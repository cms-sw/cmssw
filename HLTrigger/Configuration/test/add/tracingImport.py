"""
This is a modifed version of /Demo/imputil/knee.py from python 2.6.
After importing this module, the __import__ builtin is replaced by this
customised version:
 - modules are imported normally
 - for all instances of 'cms.Sequcne', 'cms.Path', 'cms.EndPath' their name
   are added to original_sequence, original_paths, original_endpaths, respectively, 
   in the order in which they are defined
"""

import sys, imp, __builtin__
import re

# patterns to discover cms.Path and cms.EndPath definitions in imported files
pattern_path     = re.compile(r'(\w+)\s*=\s*cms\.Path')
pattern_endpath  = re.compile(r'(\w+)\s*=\s*cms\.EndPath')
pattern_sequence = re.compile(r'(\w+)\s*=\s*cms\.Sequence')

# keep track of the original order of Paths and EndPaths
original_paths     = []
original_endpaths  = []
original_sequences = []

# replacement for __import__() as in Python 2.4 - the "level" parameter is not used
def import_hook(name, globals=None, locals=None, fromlist=None, level=-1):
    parent = determine_parent(globals)
    q, tail = find_head_package(parent, name)
    m = load_tail(q, tail)
    if not fromlist:
        return q
    if hasattr(m, "__path__"):
        ensure_fromlist(m, fromlist)
    return m

def determine_parent(globals):
    if not globals or "__name__" not in globals:
        return None
    pname = globals['__name__']
    if "__path__" in globals:
        parent = sys.modules[pname]
        assert globals is parent.__dict__
        return parent
    if '.' in pname:
        i = pname.rfind('.')
        pname = pname[:i]
        parent = sys.modules[pname]
        assert parent.__name__ == pname
        return parent
    return None

def find_head_package(parent, name):
    if '.' in name:
        i = name.find('.')
        head = name[:i]
        tail = name[i+1:]
    else:
        head = name
        tail = ""
    if parent:
        qname = "%s.%s" % (parent.__name__, head)
    else:
        qname = head
    q = import_module(head, qname, parent)
    if q: return q, tail
    if parent:
        qname = head
        parent = None
        q = import_module(head, qname, parent)
        if q: return q, tail
    raise ImportError("No module named " + qname)

def load_tail(q, tail):
    m = q
    while tail:
        i = tail.find('.')
        if i < 0: i = len(tail)
        head, tail = tail[:i], tail[i+1:]
        mname = "%s.%s" % (m.__name__, head)
        m = import_module(head, mname, m)
        if not m:
            raise ImportError("No module named " + mname)
    return m

def ensure_fromlist(m, fromlist, recursive=0):
    for sub in fromlist:
        if sub == "*":
            if not recursive:
                try:
                    all = m.__sorted__
                except AttributeError:
                    pass
                else:
                    ensure_fromlist(m, all, 1)
            continue
        if sub != "*" and not hasattr(m, sub):
            subname = "%s.%s" % (m.__name__, sub)
            submod = import_module(sub, subname, m)
            if not submod:
                raise ImportError("No module named " + subname)

def import_module(partname, fqname, parent):
    try:
        return sys.modules[fqname]
    except KeyError:
        pass
    try:
        fp, pathname, stuff = imp.find_module(partname,
                                              parent and parent.__path__)
    except ImportError:
        return None
    try:
        if fp: 
            rewind = fp.tell()
        m = imp.load_module(fqname, fp, pathname, stuff)
        if fp:
            fp.seek(rewind)
            (suffix, mode, type) = stuff
            if type == imp.PY_SOURCE:
                source = fp.read()
                for item in pattern_sequence.findall(source):
                    if item not in original_sequences: original_sequences.append( item )
                for item in pattern_path.findall(source):
                    if item not in original_paths: original_paths.append( item )
                for item in pattern_endpath.findall(source):
                    if item not in original_endpaths: original_endpaths.append( item )
            elif type == imp.PY_COMPILED:
                # can we do something about "compiled" python modules ?
                pass
    finally:
        if fp: fp.close()
    if parent:
        setattr(parent, partname, m)
    return m


# Replacement for reload()
def reload_hook(module):
    name = module.__name__
    if '.' not in name:
        return import_module(name, name, None)
    i = name.rfind('.')
    pname = name[:i]
    parent = sys.modules[pname]
    return import_module(name[i+1:], name, parent)


# Save the original hooks
original_import = __builtin__.__import__
original_reload = __builtin__.reload

# Now install our hooks
__builtin__.__import__ = import_hook
__builtin__.reload = reload_hook
