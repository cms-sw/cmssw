#!/usr/bin/env python
'''CMS Conditions DB Serialization generator.

Generates the non-intrusive serialization code required for the classes
marked with the COND_SERIALIZABLE macro.

The code was taken from the prototype that did many other things as well
(finding transients, marking serializable classes, etc.). After removing
everything but what is required to build the serialization, the code was
made more robust and cleaned up a bit to be integrated on the BoostIO IB.
However, the code still needs to be restructured a bit more to improve
readability (e.g. name some constants, use a template engine, ask for
clang's bindings to be installed along clang itself, etc.).
'''

__author__ = 'Miguel Ojeda'
__copyright__ = 'Copyright 2014, CERN'
__credits__ = ['Giacomo Govi', 'Miguel Ojeda', 'Andreas Pfeiffer']
__license__ = 'Unknown'
__maintainer__ = 'Miguel Ojeda'
__email__ = 'mojedasa@cern.ch'


import argparse
import logging
import os
import re
import subprocess

import clang.cindex


headers_template = '''
#include "{headers}"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

'''

serialize_method_begin_template = '''template <class Archive>
void {klass}::serialize(Archive & ar, const unsigned int)
{{'''

serialize_method_base_object_template = '    ar & boost::serialization::make_nvp("{base_object_name_sanitised}", boost::serialization::base_object<{base_object_name}>(*this));'

serialize_method_member_template = '''    ar & boost::serialization::make_nvp("{member_name_sanitised}", {member_name});'''

serialize_method_end = '''}
'''

instantiation_template = '''COND_SERIALIZATION_INSTANTIATE({klass});
'''


skip_namespaces = frozenset([
    # Do not go inside anonymous namespaces (static)
    '',

    # Do not go inside some standard namespaces
    'std', 'boost', 'mpl_', 'boost_swap_impl',

    # Do not go inside some big namespaces coming from externals
    'ROOT', 'edm', 'ora', 'coral', 'CLHEP', 'Geom', 'HepGeom',
])

def is_definition_by_loc(node):
    if node.get_definition() is None:
        return False
    if node.location is None or node.get_definition().location is None:
        return False
    return node.location == node.get_definition().location

def is_serializable_class(node):
    for child in node.get_children():
        if child.spelling != 'serialize' or child.kind != clang.cindex.CursorKind.FUNCTION_TEMPLATE or is_definition_by_loc(child):
            continue

        if [(x.spelling, x.kind, is_definition_by_loc(x), x.type.kind) for x in child.get_children()] != [
            ('Archive', clang.cindex.CursorKind.TEMPLATE_TYPE_PARAMETER, True, clang.cindex.TypeKind.UNEXPOSED),
            ('ar', clang.cindex.CursorKind.PARM_DECL, True, clang.cindex.TypeKind.LVALUEREFERENCE),
            ('version', clang.cindex.CursorKind.PARM_DECL, True, clang.cindex.TypeKind.UINT),
        ]:
            continue

        return True

    return False


def is_serializable_class_manual(node):
    for child in node.get_children():
        if child.spelling == 'cond_serialization_manual' and child.kind == clang.cindex.CursorKind.CXX_METHOD and not is_definition_by_loc(child):
            return True

    return False


def get_statement(node):
    # For some cursor kinds, their location is empty (e.g. translation units
    # and attributes); either because of a bug or because they do not have
    # a meaningful 'start' -- however, the extent is always available
    if node.extent.start.file is None:
        return None

    filename = node.extent.start.file.name
    start = node.extent.start.offset
    end = node.extent.end.offset

    with open(filename, 'rb') as fd:
        source = fd.read()

    return source[start:source.find(';', end)]


def get_basic_type_string(node):
    typekinds = {
        clang.cindex.TypeKind.BOOL: 'bool',
        clang.cindex.TypeKind.INT: 'int',
        clang.cindex.TypeKind.LONG: 'long',
        clang.cindex.TypeKind.UINT: 'unsigned int',
        clang.cindex.TypeKind.ULONG: 'unsigned long',
        clang.cindex.TypeKind.FLOAT: 'float',
        clang.cindex.TypeKind.DOUBLE: 'double',
    }

    if node.type.kind not in typekinds:
        raise Exception('Not a known basic type.')

    return typekinds[node.type.kind]


def get_type_string(node):
    spelling = node.type.get_declaration().spelling
    if spelling is not None:
        return spelling

    return get_basic_type_string(node)


def get_serializable_classes_members(node, all_template_types=None, namespace='', only_from_path=None):
    if all_template_types is None:
        all_template_types = []

    logging.debug('%s', (node.spelling, all_template_types, namespace))
    results = {}
    for child in node.get_children():
        if child.kind == clang.cindex.CursorKind.NAMESPACE:
            # If we are in the root namespace, let's skip some common, big
            # namespaces to improve speed and avoid serializing those.
            if namespace == '':
                if child.spelling in skip_namespaces:
                    continue

                # This skips compiler-specific stuff as well (e.g. __gnucxx...)
                if child.spelling.startswith('_'):
                    continue

            logging.debug('Going into namespace %s', child.spelling)

            results.update(get_serializable_classes_members(child, all_template_types, namespace + child.spelling + '::', only_from_path))
            continue

        if child.kind in [clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL, clang.cindex.CursorKind.CLASS_TEMPLATE] and is_definition_by_loc(child):
            logging.debug('Found struct/class/template definition: %s', child.spelling if child.spelling else '<anonymous>')

            if only_from_path is not None \
                and child.location.file is not None \
                and not child.location.file.name.startswith(only_from_path):
                    logging.debug('Skipping since it is an external of this package: %s', child.spelling)
                    continue

            serializable = is_serializable_class(child)
            if serializable:
                if child.spelling == '':
                    raise Exception('It is not possible to serialize anonymous/unnamed structs/classes.')

                if is_serializable_class_manual(child):
                    logging.info('Found manual serializable struct/class/template: %s', child.spelling)
                    continue

                logging.info('Found serializable struct/class/template: %s', child.spelling)

            template_types = []
            base_objects = []
            members = []
            transients = []
            after_serialize = False
            after_serialize_count = 0
            for member in child.get_children():
                if after_serialize:
                    if after_serialize_count == 2:
                        after_serialize = False
                    else:
                        after_serialize_count = after_serialize_count + 1

                        if member.kind != clang.cindex.CursorKind.UNEXPOSED_DECL:
                            raise Exception('Expected unexposed declaration (friend) after serialize() but found something else: looks like the COND_SERIALIZABLE macro has been changed without updating the script.')

                        if 'COND_SERIALIZABLE' not in get_statement(member):
                            raise Exception('Could not find COND_SERIALIZABLE in the statement of the expected unexposed declarations (friends) after serialize(). Please fix the script/macro.')

                        logging.debug('Skipping expected unexposed declaration (friend) after serialize().')
                        continue

                # Template type parameters (e.g. <typename T>)
                if member.kind == clang.cindex.CursorKind.TEMPLATE_TYPE_PARAMETER:
                    logging.info('    Found template type parameter: %s', member.spelling)
                    template_types.append(('typename', member.spelling))

                # Template non-type parameters (e.g. <int N>)
                elif member.kind == clang.cindex.CursorKind.TEMPLATE_NON_TYPE_PARAMETER:
                    type_string = get_type_string(member)
		    if not type_string: 
		       type_string = get_basic_type_string(member)
                    logging.info('    Found template non-type parameter: %s %s', type_string, member.spelling)
                    template_types.append((type_string, member.spelling))

                # Base objects
                elif member.kind == clang.cindex.CursorKind.CXX_BASE_SPECIFIER:
                    # FIXME: .displayname gives sometimes things like "class mybase"
                    base_object = member.displayname
                    prefix = 'class '
                    if base_object.startswith(prefix):
                        base_object = base_object[len(prefix):]
                    logging.info('    Found base object: %s', base_object)
                    base_objects.append(base_object)

                # Member variables
                elif member.kind == clang.cindex.CursorKind.FIELD_DECL and is_definition_by_loc(member):
                    # While clang 3.3 does not ignore unrecognized attributes
                    # (see http://llvm.org/viewvc/llvm-project?revision=165082&view=revision )
                    # for some reason they do not appear in the bindings yet
                    # so we just do it ourselves.

                    # FIXME: To simplify and avoid parsing C++ ourselves, our transient
                    # attribute applies to *all* the variables declared in the same statement.
                    if 'COND_TRANSIENT' not in get_statement(member):
                        logging.info('    Found member variable: %s', member.spelling)
                        members.append(member.spelling)
                    else:
                        if serializable:
                            logging.info('    Found transient member variable: %s', member.spelling)
                            transients.append(member.spelling)
                        else:
                            raise Exception('Transient %s found for non-serializable class %s', member.spelling, child.spelling)

                elif member.kind == clang.cindex.CursorKind.FUNCTION_TEMPLATE and member.spelling == 'serialize':
                    after_serialize = True
                    logging.debug('Found serialize() method, skipping next two children which must be unexposed declarations.')

                elif member.kind in frozenset([
                    # For safety, we list all known kinds that we need to skip
                    # and raise in unknown cases (this helps catching problems
                    # with undefined classes)
                    clang.cindex.CursorKind.CONSTRUCTOR,
                    clang.cindex.CursorKind.DESTRUCTOR,
                    clang.cindex.CursorKind.CXX_METHOD,
                    clang.cindex.CursorKind.CXX_ACCESS_SPEC_DECL,
                    clang.cindex.CursorKind.FUNCTION_TEMPLATE,
                    clang.cindex.CursorKind.TYPEDEF_DECL,
                    clang.cindex.CursorKind.CLASS_DECL,
                    clang.cindex.CursorKind.ENUM_DECL,
                    clang.cindex.CursorKind.VAR_DECL,
                    clang.cindex.CursorKind.STRUCT_DECL,
                    clang.cindex.CursorKind.UNION_DECL,
                    clang.cindex.CursorKind.CONVERSION_FUNCTION,
                    clang.cindex.CursorKind.TYPE_REF,
                    clang.cindex.CursorKind.DECL_REF_EXPR,
                ]):
                    logging.debug('Skipping member: %s %s %s %s', member.displayname, member.spelling, member.kind, member.type.kind)

                elif member.kind == clang.cindex.CursorKind.UNEXPOSED_DECL:
                    statement = get_statement(member)

                    # Friends are unexposed but they are not data to serialize
                    if 'friend' in statement:
                        # If we know about them, skip the warning
                        if \
                            'friend class ' in statement or \
                            'friend struct ' in statement or \
                            'friend std::ostream& operator<<(' in statement or \
                            'friend std::istream& operator>>(' in statement:
                            logging.debug('Skipping known friend: %s', statement.splitlines()[0])
                            continue

                        # Otherwise warn
                        logging.warning('Unexposed declaration that looks like a friend declaration -- please check: %s %s %s %s %s', member.displayname, member.spelling, member.kind, member.type.kind, statement)
                        continue

                    raise Exception('Unexposed declaration. This probably means (at the time of writing) that an unknown class was found (may happen, for instance, when the compiler does not find the headers for std::vector, i.e. missing -I option): %s %s %s %s %s' % (member.displayname, member.spelling, member.kind, member.type.kind, statement))

                else:
                    raise Exception('Unknown kind. Please fix the script: %s %s %s %s %s' % (member.displayname, member.spelling, member.kind, member.type.kind, statement))

            if template_types:
                template_use = '%s<%s>' % (child.spelling, ', '.join([template_type_name for (_, template_type_name) in template_types]))
            else:
                template_use = child.spelling

            new_namespace = namespace + template_use

            new_all_template_types = all_template_types + [template_types]

            results[new_namespace] = (child, serializable, new_all_template_types, base_objects, members, transients)

            results.update(get_serializable_classes_members(child, new_all_template_types, new_namespace + '::', only_from_path))

    for (klass, (node, serializable, all_template_types, base_objects, members, transients)) in results.items():
        if serializable and len(members) == 0:
            logging.info('No non-transient members found for serializable class %s', klass)

    return results


def split_path(path):
    folders = []

    while True:
        path, folder = os.path.split(path)

        if folder != '':
            folders.append(folder)
        else:
            if path != '':
                folders.append(path)
            break

    folders.reverse()

    return folders


def get_flags(product_name, flags):
    command = "scram b echo_%s_%s | tail -1 | cut -d '=' -f '2-' | xargs -n1" % (product_name, flags)
    logging.debug('Running: %s', command)
    return subprocess.check_output(command, shell=True).splitlines()

def log_flags(name, flags):
    logging.debug('%s = [', name)
    for flag in flags:
        logging.debug('    %s', flag)
    logging.debug(']')


def get_diagnostics(translation_unit):
    return map(lambda diag: {
        'severity' : diag.severity,
        'location' : diag.location,
        'spelling' : diag.spelling,
        'ranges' : diag.ranges,
        'fixits' : diag.fixits,
    }, translation_unit.diagnostics)


def get_default_gcc_search_paths(gcc = 'g++', language = 'c++'):
    command = 'echo "" | %s -x%s -v -E - 2>&1' % (gcc, language)
    logging.debug('Running: %s', command)

    paths = []
    in_list = False
    for line in subprocess.check_output(command, shell=True).splitlines():
        if in_list:
            if line == 'End of search list.':
                break

            path = os.path.normpath(line.strip())

            # Intrinsics not handled by clang
            # Note that /lib/gcc is found in other paths if not normalized,
            # so has to go after normpath()
            if '/lib/gcc/' in path:
                continue

            paths.append('-I%s' % path)

        else:
            if line == '#include <...> search starts here:':
                in_list = True

    if not in_list:
        raise Exception('Default GCC search paths not found.')

    return paths

def sanitise(var):
    return re.sub('[^a-zA-Z0-9.,-:]', '-', var)


class SerializationCodeGenerator(object):

    def __init__(self, scramFlags=None):

        self.cmssw_base = os.getenv('CMSSW_BASE')
        if self.cmssw_base is None:
            raise Exception('CMSSW_BASE is not set.')
        logging.debug('cmssw_base = %s', self.cmssw_base)

        cwd = os.getcwd()
        logging.debug('cwd = %s', cwd)

        if not cwd.startswith(self.cmssw_base):
            raise Exception('The filepath does not start with CMSSW_BASE.')

        relative_path = cwd[len(self.cmssw_base)+1:]
        logging.debug('relative_path = %s', relative_path)

        self.split_path = split_path(relative_path)
        logging.debug('splitpath = %s', self.split_path)

        if len(self.split_path) < 3:
            raise Exception('This script requires to be run inside a CMSSW package (usually within CondFormats), e.g. CondFormats/Alignment. The current path is: %s' % self.split_path)

        if self.split_path[0] != 'src':
            raise Exception('The first folder should be src.')

        if self.split_path[1] != 'CondFormats':
            raise Exception('The second folder should be CondFormats.')

        product_name = '%s%s' % (self.split_path[1], self.split_path[2])
        logging.debug('product_name = %s', product_name)

	if not scramFlags:
	   cpp_flags = get_flags(product_name, 'CPPFLAGS')
           cxx_flags = get_flags(product_name, 'CXXFLAGS')
	else:
	   cpp_flags = self.cleanFlags( scramFlags )
	   cxx_flags = []

        # We are using libClang, thus we have to follow Clang include paths
        std_flags = get_default_gcc_search_paths(gcc='clang++')
        log_flags('cpp_flags', cpp_flags)
        log_flags('cxx_flags', cxx_flags)
        log_flags('std_flags', std_flags)

        flags = ['-xc++'] + cpp_flags + cxx_flags + std_flags

        headers_h = self._join_package_path('src', 'headers.h')
        logging.debug('headers_h = %s', headers_h)
        if not os.path.exists(headers_h):
            raise Exception('File %s does not exist. Impossible to serialize package.' % headers_h)

        logging.info('Searching serializable classes in %s/%s ...', self.split_path[1], self.split_path[2])

        logging.debug('Parsing C++ classes in file %s ...', headers_h)
        index = clang.cindex.Index.create()
        translation_unit = index.parse(headers_h, flags)
        if not translation_unit:
            raise Exception('Unable to load input.')

        severity_names = ('Ignored', 'Note', 'Warning', 'Error', 'Fatal')
        get_severity_name = lambda severity_num: severity_names[severity_num] if severity_num < len(severity_names) else 'Unknown'
        max_severity_level = 0 # Ignored
        diagnostics = get_diagnostics(translation_unit)
        for diagnostic in diagnostics:
            logf = logging.error

            # Ignore some known warnings
            if diagnostic['spelling'].startswith('argument unused during compilation') \
                or diagnostic['spelling'].startswith('unknown warning option'):
                logf = logging.debug

            logf('Diagnostic: [%s] %s', get_severity_name(diagnostic['severity']), diagnostic['spelling'])
            logf('   at line %s in %s', diagnostic['location'].line, diagnostic['location'].file)

            max_severity_level = max(max_severity_level, diagnostic['severity'])

        if max_severity_level >= 3: # Error
            raise Exception('Please, resolve all errors before proceeding.')

        self.classes = get_serializable_classes_members(translation_unit.cursor, only_from_path=self._join_package_path())

    def _join_package_path(self, *path):
        return os.path.join(self.cmssw_base, self.split_path[0], self.split_path[1], self.split_path[2], *path)

    def cleanFlags(self, flagsIn):
	flags = [ flag for flag in flagsIn if not flag.startswith(('-march', '-mtune', '-fdebug-prefix-map')) ]
        blackList = ['--', '-fipa-pta']
        return [x for x in flags if x not in blackList]

    def generate(self, outFileName):

    	filename = outFileName
	if not filename:  # in case we're not using scram, this may not be set, use the default then, assuming we're in the package dir ...
	   filename = self._join_package_path('src', 'Serialization.cc')

        n_serializable_classes = 0

        source = headers_template.format(headers=os.path.join(self.split_path[1], self.split_path[2], 'src', 'headers.h'))

        for klass in sorted(self.classes):
            (node, serializable, all_template_types, base_objects, members, transients) = self.classes[klass]

            if not serializable:
                continue

            n_serializable_classes += 1

            skip_instantiation = False
            for template_types in all_template_types:
                if template_types:
                    skip_instantiation = True
                    source += ('template <%s>' % ', '.join(['%s %s' % template_type for template_type in template_types])) + '\n'

            source += serialize_method_begin_template.format(klass=klass) + '\n'

            for base_object_name in base_objects:
                base_object_name_sanitised = sanitise(base_object_name)
                source += serialize_method_base_object_template.format(base_object_name=base_object_name, base_object_name_sanitised=base_object_name_sanitised) + '\n'

            for member_name in members:
                member_name_sanitised = sanitise(member_name)
                source += serialize_method_member_template.format(member_name=member_name, member_name_sanitised=member_name_sanitised) + '\n'

            source += serialize_method_end

            if skip_instantiation:
                source += '\n'
            else:
                source += instantiation_template.format(klass=klass) + '\n'

        if n_serializable_classes == 0:
            raise Exception('No serializable classes found, while this package has a headers.h file.')

        # check if we have a file for template instantiations and other "special" code:
        if os.path.exists( './src/SerializationManual.h' ) :
            source += '#include "%s/%s/src/SerializationManual.h"\n' % (self.split_path[1], self.split_path[2])

        logging.info('Writing serialization code for %s classes in %s ...', n_serializable_classes, filename)
        with open(filename, 'wb') as fd:
            fd.write(source)


def main():
    parser = argparse.ArgumentParser(description='CMS Condition DB Serialization generator.')
    parser.add_argument('--verbose', '-v', action='count', help='Verbosity level. -v reports debugging information.')
    parser.add_argument('--output' , '-o', action='store', help='Specifies the path to the output file written. Default: src/Serialization.cc')
    parser.add_argument('--package', '-p', action='store', help='Specifies the path to the package to be processed. Default: the actual package')

    opts, args = parser.parse_known_args()

    logLevel = logging.INFO
    if opts.verbose < 1 and opts.output and opts.package:   # assume we're called by scram and reduce logging - but only if no verbose is requested
       logLevel = logging.WARNING

    if opts.verbose >= 1: 
       logLevel = logging.DEBUG

    logging.basicConfig(
        format = '[%(asctime)s] %(levelname)s: %(message)s',
        level = logLevel,
    )

    if opts.package:  # we got a directory name to process, assume it's from scram and remove the last ('/src') dir from the path
        pkgDir = opts.package
	if pkgDir.endswith('/src') :
	    pkgDir, srcDir = os.path.split( opts.package )
        os.chdir( pkgDir )
	logging.info("Processing package in %s " % pkgDir)

    if opts.output:
       logging.info("Writing serialization code to %s " % opts.output)

    SerializationCodeGenerator( scramFlags=args[1:] ).generate( opts.output )

if __name__ == '__main__':
    main()

