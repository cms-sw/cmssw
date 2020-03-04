#!/usr/bin/env python


from __future__ import print_function
import shutil
import glob
import json
import yaml
import sys
import sys
import os

from importlib  import import_module
from argparse   import ArgumentParser, RawTextHelpFormatter


import pluginCondDBV2PyInterface
pluginCondDBV2PyInterface.CMSSWInit()



def supress_output( f ):
    '''
    Temporarily disables stdout and stderr so that printouts from the plot
    plugin does not compromise the purity of our ssh stream if 
    args.suppress_output is true
    '''
    def decorated( *fargs, **fkwargs ):

        suppress = args.suppress_output
        if suppress:
            
            # Get rid of what is already there ( should be nothing for this script )
            sys.stdout.flush()

            # Save file descriptors so it can be reactivated later 
            saved_stdout = os.dup( 1 )
            saved_stderr = os.dup( 2 )

            # /dev/null is used just to discard what is being printed
            devnull = os.open( '/dev/null', os.O_WRONLY )

            # Duplicate the file descriptor for /dev/null
            # and overwrite the value for stdout (file descriptor 1)
            os.dup2( devnull, 1 )
            os.dup2( devnull, 2 )

        result = f( *fargs, **fkwargs )

        if suppress:

            # Close devnull after duplication (no longer needed)
            os.close( devnull )

            # Reenable stdout and stderr
            os.dup2( saved_stdout, 1 )
            os.dup2( saved_stderr, 2 )

        return result

    return decorated


@supress_output
def deserialize_iovs(db, plugin_name, plot_name, tag, time_type, iovs, input_params):
    ''' Deserializes given iovs data and returns plot coordinates '''
    
    output('Starting to deserialize iovs: ', '')
    output('db: ', db)
    output('plugin name: ', plugin_name)
    output('plot name: ', plot_name)
    output('tag name: ', tag)
    output('tag time type: ', time_type)
    output('iovs: ', iovs)
  
    plugin_base = import_module('pluginModule_PayloadInspector')
    output('PI plugin base: ', plugin_base)

    plugin_obj = import_module(plugin_name)
    output('PI plugin object: ', plugin_obj)

    # get plot method and execute it with given iovs
    plot = getattr(plugin_obj, plot_name)()
    output('plot object: ', plot)

    if db == "Prod":
        db_name = 'frontier://FrontierProd/CMS_CONDITIONS'
    elif db == 'Prep' :
        db_name = 'frontier://FrontierPrep/CMS_CONDITIONS'
    else:
        db_name = db

    output('full DB name: ', db_name)

    if input_params is not None:
        plot.setInputParamValues( input_params )
    success = plot.process(db_name, tag, time_type, int(iovs['start_iov']), int(iovs['end_iov']))
    output('plot processed data successfully: ', success)
    if not success:
        return False


    result = plot.data()
    output('deserialized data: ', result)
    return result
@supress_output
def deserialize_twoiovs(db, plugin_name, plot_name, tag,tagtwo,iovs,iovstwo, input_params):
    ''' Deserializes given iovs data and returns plot coordinates '''
    #print "Starting to deserialize iovs:"
    #print 'First Iovs',iovs
    #print 'Two Iovs', iovstwo
    output('Starting to deserialize iovs: ', '')
    output('db: ', db)
    output('plugin name: ', plugin_name)
    output('plot name: ', plot_name)
    output('tag name: ', tag)
    output('tagtwo name: ', tagtwo)
    #output('tag time type: ', time_type)
    output('iovs: ', iovs)
    output('iovstwo: ', iovstwo)
  
    plugin_base = import_module('pluginModule_PayloadInspector')
    output('PI plugin base: ', plugin_base)

    plugin_obj = import_module(plugin_name)
    output('PI plugin object: ', plugin_obj)

    # get plot method and execute it with given iovs
    plot = getattr(plugin_obj, plot_name)()
    output('plot object: ', plot)

    db_name = 'oracle://cms_orcon_adg/CMS_CONDITIONS' if db == 'Prod' else 'oracle://cms_orcoff_prep/CMS_CONDITIONS'
    output('full DB name: ', db_name)

    if input_params is not None:
        plot.setInputParamValues( input_params )
    success = plot.processTwoTags(db_name, tag,tagtwo,int(iovs['start_iov']), int(iovstwo['end_iov']))
    #print "All good",success
    output('plot processed data successfully: ', success)
    if not success:
        return False


    result = plot.data()
    output('deserialized data: ', result)
    return result

def discover_plugins():
    ''' Returns a list of Payload Inspector plugin names
        Example:
        ['pluginBasicPayload_PayloadInspector', 'pluginBeamSpot_PayloadInspector', 'pluginSiStrip_PayloadInspector']
    '''
    architecture = os.environ.get('SCRAM_ARCH', None)
    output('architecture: ', architecture)

    plugins = []
    releases = [
        os.environ.get('CMSSW_BASE', None),
        os.environ.get('CMSSW_RELEASE_BASE', None)
    ]

    for r in releases:
        if not r: continue # skip if release base is not specified
        output('* release: ', r)

        path = os.path.join(r, 'lib', architecture)
        output('* full release path: ', path)

        plugins += glob.glob(path + '/plugin*_PayloadInspector.so' )
        output('found plugins: ', plugins) 
        
        if r: break # break loop if CMSSW_BASE is specified        
  
    # extracts the object name from plugin path:
    # /afs/cern.ch/cms/slc6_amd64_gcc493/cms/cmssw/CMSSW_8_0_6/lib/slc6_amd64_gcc493/pluginBasicPayload_PayloadInspector.so
    # becomes pluginBasicPayload_PayloadInspector
    result = []
    for p in plugins:
         result.append(p.split('/')[-1].replace('.so', ''))

    output('discovered plugins: ', result)
    return result

def discover():
    ''' Discovers object types and plots for a given cmssw release
        Example:
        {
            "BasicPayload": [
                {"plot": "plot_BeamSpot_x", "plot_type": "History", 
                 "single_iov": false, "plugin_name": "pluginBeamSpot_PayloadInspector",
                 "title": "x vs run number"},
                ...
            ],
           ...
        }
    '''
    plugin_base = import_module('pluginModule_PayloadInspector') 
    result = {}
    for plugin_name in discover_plugins():
        output(' - plugin name: ', plugin_name)
        plugin_obj = import_module(plugin_name)
        output('*** PI plugin object: ', plugin_obj)
        for plot in dir(plugin_obj):
            if 'plot_' not in plot: continue # skip if method doesn't start with 'plot_' prefix
            output(' - plot name: ', plot)
            plot_method= getattr(plugin_obj, plot)()
            output(' - plot object: ', plot_method)
            payload_type = plot_method.payloadType()
            output(' - payload type: ', payload_type)
            plot_title = plot_method.title()
            output(' - plot title: ', plot_title)
            plot_type = plot_method.type()
            output(' - plot type: ', plot_type)
            single_iov = plot_method.isSingleIov()
            output(' - is single iov: ', single_iov)
            two_tags = plot_method.isTwoTags()
            output(' - is Two Tags: ', two_tags)
            input_params = plot_method.inputParams()
            output(' - input params: ', len(input_params))
            result.setdefault(payload_type, []).append({'plot': plot, 'plugin_name': plugin_name, 'title': plot_title, 'plot_type': plot_type, 'single_iov': single_iov, 'two_tags': two_tags, 'input_params': input_params})
            output('currently discovered info: ', result)
    output('*** final output:', '')
    return json.dumps(result)

def output(description, param):
    if args.verbose:
        print('')
        print(description, param)

if __name__ == '__main__':

    description = '''
    Payload Inspector - data visualisation tool which is integrated into the cmsDbBrowser.
    It allows to display plots and monitor the calibration and alignment data.

    You can access Payload Inspector with a link below:
    https://cms-conddb.cern.ch/cmsDbBrowser/payload_inspector/Prod

    This script is a part of the Payload Inspector service and is responsible for:
    a) discovering PI objects that are available in a given cmssw release
    b) deserializing payload data which is later used as plot coordinates
    c) testing new PI objects which are under development

    To test new PI objects please do the following:
    a) run ./getPayloadData.py --discover
    to check if your newly created object is found by the script.
    Please note that we strongly rely on naming conventions so if you don't
    see your object listed you probably misnamed it in objectType() method.
    Also all plot methods should start with "plot_" prefix.

    b) second step is to test if it returns data correctly:
    run ./getPayloadData.py --plugin YourPIPluginName --plot YourObjectPlot --tag tagName --time_type Run --iovs '{"start_iov": "201", "end_iov": "801"}' --db Prod --test 
  
    Here is an example for BasicPayload object:
    run ./getPayloadData.py --plugin pluginBasicPayload_PayloadInspector --plot plot_BasicPayload_data0 --tag BasicPayload_v2 --time_type Run --iovs '{"start_iov": "201", "end_iov": "801"}' --db Prod --test

    c) if it works correctly please make a pull request and once it's accepted
    go to cmsDbBrowser and wait for the next IB to test it.
    '''

    parser = ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument("-d", "--discover",   help="discovers object types and plots \nfor a given cmssw release", action="store_true")
    parser.add_argument("-i", "--iovs",       help="deserializes given iovs data encoded in base64 and returns plot coordinates also encoded in base64")
    parser.add_argument("-i2", "--iovstwo",   help="deserializes given iovs data encoded in base64 and returns plot coordinates also encoded in base64")
    parser.add_argument("-o", "--plugin",     help="Payload Inspector plugin name needed for iovs deserialization")
    parser.add_argument("-p", "--plot",       help="plot name needed for iovs deserialization")
    parser.add_argument("-t", "--tag",        help="tag name needed for iovs deserialization")
    parser.add_argument("-t2", "--tagtwo",    help="tag name needed for iovs deserialization")
    parser.add_argument("-tt", "--time_type", help="tag time type name needed for iovs deserialization")
    parser.add_argument("-b", "--db",         help="db (Prod or Prep) needed for iovs deserialization")
    parser.add_argument("-test", "--test",    help="add this flag if you want to test the deserialization function and want to see a readable output", action="store_true")
    parser.add_argument("-v", "--verbose",    help="verbose mode. Shows more information", action="store_true")
    parser.add_argument("-ip","--image_plot", help="Switch telling the script that this plot type is of type Image", action="store_true")
    parser.add_argument("-s", "--suppress-output", help="Supresses output from so that stdout and stderr can be kept pure for the ssh transmission", action="store_true")
    parser.add_argument("-is", "--input_params", help="Plot input parameters ( dictionary, JSON serialized into string )" )

    # shows help if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    # Return discover of plot if requested
    if args.discover:
        os.write( 1, discover() )

    input_params = None
    if args.input_params is not None:
        input_params = yaml.safe_load(args.input_params)

    # Return a plot if iovs are provided
    #print '* getiovs: ',args.iovs
    #print '* getiovstwo: ',args.iovstwo
    if args.iovstwo:

        # Run plugin with arguments
        #print 'We are here'
        a=json.loads(args.iovs)
        #print 'A',a
        b=json.loads(args.iovstwo)
        #print 'B',b
        result = deserialize_twoiovs(args.db, args.plugin, args.plot, args.tag,args.tagtwo,a,b, input_params)
        # If test -> output the result as formatted json
        if args.test:
            os.write( 1, json.dumps( json.loads( result ), indent=4 ))
        #print 'Result:',result
	if args.image_plot:
            try:
                filename = json.loads( result )['file']
                #print 'File name',filename
            except ValueError as e:
                os.write( 2, 'Value error when getting image name: %s\n' % str( e ))
            except KeyError as e:
                os.write( 2, 'Key error when getting image name: %s\n' % str( e ))

            if not filename or not os.path.isfile( filename ):
                os.write( 2, 'Error: Generated image file (%s) not found\n' % filename )

            try:
                with open( filename, 'r' ) as f:
                    shutil.copyfileobj( f, sys.stdout )
            except IOError as e:
                os.write( 2, 'IO error when streaming image: %s' % str( e ))
            finally:
                os.remove( filename )

                        
        # Else -> output result json string with base 64 encoding
    elif args.iovs:
        result = deserialize_iovs(args.db, args.plugin, args.plot, args.tag, args.time_type, json.loads(args.iovs), input_params)
        
        # If test -> output the result as formatted json
        if args.test:
            os.write( 1, json.dumps( json.loads( result ), indent=4 ))

        # If image plot -> get image file from result, open it and output bytes 
        elif args.image_plot:

            filename = None
            
            try:
                filename = json.loads( result )['file']
                #print 'File name',filename
            except ValueError, e:
                os.write( 2, 'Value error when getting image name: %s\n' % str( e ))
            except KeyError, e:
                os.write( 2, 'Key error when getting image name: %s\n' % str( e ))

            if not filename or not os.path.isfile( filename ):
                os.write( 2, 'Error: Generated image file (%s) not found\n' % filename )

            try:
                with open( filename, 'r' ) as f:
                    shutil.copyfileobj( f, sys.stdout )
            except IOError, e:
                os.write( 2, 'IO error when streaming image: %s' % str( e ))
            finally:
                os.remove( filename )

                        
        # Else -> output result json string with base 64 encoding
        else: 
            os.write( 1, result.encode( 'base64' ))

