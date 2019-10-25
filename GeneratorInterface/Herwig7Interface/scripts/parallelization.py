#!/usr/bin/env python

# This script sets up parallel jobs for the build, integrate and run
# step when using Herwig with the CMSSW framework.
# It takes a cmsRun file, adjusts the parameters in it accordingly to
# the options and saves them to temporary cmsRun files. For each step
# a different cmsRun file is created. The original file remains
# unaltered.

# Possible options:
# -b/--build : sets the number of build jobs and starts the build step.
# -i/--integrate : sets the maximal number of integration jobs
#     This option already has to be set when the build step is invoked.
#     The integration step will be performed if this option is set,
#     unless --nointegration is chosen.
#     The actual number of integration jobs may be smaller. It is
#     determined by the number of files in Herwig-scratch/Build.
# -r/--run : sets the number of run jobs and starts the run step.
# --nointegration : use this option to set up several integration jobs
#     without actually performing them
# --stoprun: use this option if you want to create the cmsRun files
#     without calling cmsRun
# --resumerun: no new cmsRun files for the run step will be created
#     For this option to work 'temporary' cmsRun files complying to the
#     naming scheme have to be availible. Only files up to the number
#     of jobs defined by --run will be considered.
# --keepfiles : don't remove the created temporary cmsRun files
# --l/--log: write the output of each shell command called in a
#     seperate log file

# Comments in the cmsRun file in the process.generator part may confuse
# this script. Check the temporary cmsRun files if errors occur.

# A parallelized run step is achieved by calling cmsRun an according
# number of times with different seeds for Herwig. The built in feature
# of Herwig wont be used.

# Author: Dominik Beutel


from __future__ import print_function
import argparse
import sys
import os
import subprocess
import re



def uint(string):
    """Unsigned int type"""
    value = int(string)
    if value < 0:
        msg = '{0} is negative'.format(string)
        raise argparse.ArgumentTypeError(msg)
    return value



def adjust_pset(cmsrunfilename, savefilename, par_list):
    """Takes the cmsRun filem, removes all occurences of runMode, jobs,
       maxJobs and integrationList parameters in the process.generator
       part.
       The the parameters in par_list are set instead and saved.
    """ 

    with open(cmsrunfilename, 'r') as readfile:
        parsestring = readfile.read()

        # get first opening bracket after process.generator
        begin_gen_step = parsestring.find('(', parsestring.find('process.generator'))

        # find matching bracket
        end_gen_step = begin_gen_step
        bracket_counter = 1
        for position in range(begin_gen_step+1, len(parsestring)):
            if parsestring[position] == '(':
                bracket_counter += 1
            if parsestring[position] == ')':
                bracket_counter -= 1
            if not bracket_counter:
                end_gen_step = position
                break

        # get string between brackets
        gen_string = parsestring[begin_gen_step+1:end_gen_step]

        # remove all parameters that would interfere
        gen_string = re.sub(r',\s*runModeList\s*=\s*cms.untracked.string\((.*?)\)', '', gen_string)
        gen_string = re.sub(r',\s*jobs\s*=\s*cms.untracked.int32\((.*?)\)', '', gen_string)
        gen_string = re.sub(r',\s*integrationList\s*=\s*cms.untracked.string\((.*?)\)', '', gen_string)
        gen_string = re.sub(r',\s*maxJobs\s*=\s*cms.untracked.uint32\((.*?)\)', '', gen_string)
        gen_string = re.sub(r',\s*seed\s*=\s*cms.untracked.int32\((.*?)\)', '', gen_string)


    # write the savefile with all parameters given in par_list
    with open(savefilename,'w') as savefile:
        savefile.write(parsestring[:begin_gen_step+1])
        savefile.write(gen_string)
        for item in par_list:
            savefile.write(',\n')
            savefile.write(item)
        savefile.write(parsestring[end_gen_step:])



def cleanupandexit(filelist):
    """Delete the files in filelist and exit"""
    for filename in filelist:
        os.remove(filename)
    sys.exit(0)




##################################################
# Get command line arguments
##################################################

parser = argparse.ArgumentParser()

parser.add_argument('cmsRunfile', help='filename of the cmsRun configuration')
parser.add_argument('-b', '--build', help='set the number of build jobs', type=int, choices=range(0,11), default=0)
parser.add_argument('-i', '--integrate', help='set the maximal number of integration jobs', type=uint, default=0)
parser.add_argument('-r', '--run', help='set the number of run jobs', type=int, choices=range(0,11), default=0)
parser.add_argument('--nointegration', help='build -i integration jobs without actually integrating', action='store_true')
parser.add_argument('--keepfiles', help='don\'t delete temporary files', action='store_true')
parser.add_argument('--stoprun', help='stop after creating the cmsRun files for the run step', action='store_true')
parser.add_argument('--resumerun', help='use existing \'temporary\' files for the run step', action='store_true')
parser.add_argument('-l', '--log', help='write the output of each process in a separate log file', action='store_true')

args = parser.parse_args()

# List of files needed for clean-up
cleanupfiles = []

# Create a template name for all created files
template_name = args.cmsRunfile.replace('.', '_')



##################################################
# Execute the different run modes
##################################################

## Build ##

# jobs defines number of build jobs in the cmsRun file
# maxJobs tells Herwig to prepare the according number
#     of integrations

if args.build != 0:
    # Set up parameters
    parameters = ['runModeList = cms.untracked.string(\'build\')']
    parameters.append('jobs = cms.untracked.int32(' + str(args.build) + ')')
    if args.integrate != 0:
        parameters.append('maxJobs = cms.untracked.uint32(' + str(args.integrate) + ')')

    build_name = template_name + '_build.py'
    adjust_pset(args.cmsRunfile, build_name, parameters)

    cleanupfiles.append(build_name)

    # Start build job
    print('Setting up {0} build jobs.'.format(str(args.build)))
    print('Setting up a maximum of {0} integration jobs.'.format(str(args.integrate)))
    print('Calling\t\'cmsRun ' + build_name + '\'')

    if args.log:
        print('Writing ouput to log file: ' + build_name[:-2] + 'log')
        with open(build_name[:-2] + 'log', 'w') as build_log:
            process = subprocess.Popen(['cmsRun', build_name], stdout=build_log, stderr=subprocess.STDOUT)
    else:
        process = subprocess.Popen(['cmsRun ' + build_name], shell=True)
    process.wait()

    print('--------------------')
    print('Build step finished.')
    print('--------------------')



## Integrate ##

# Stop in case no integration is desired
if args.nointegration:
    print('--nointegration: Run will be stopped here.')
    cleanupandexit(cleanupfiles)

if args.integrate != 0:
    # Determine number of integration jobs
    actual_int_jobs = len([string for string in os.listdir('Herwig-scratch/Build') if re.match(r'integrationJob[0-9]+', string)])
    
    # Stop if this number exceeds the given parameter
    if actual_int_jobs > args.integrate:
        print('Actual number of integration jobs {0} exceeds \'--integrate {1}\'.'.format(actual_int_jobs, args.integrate))
        print('Integration will not be performed.')
        cleanupandexit(cleanupfiles)

    # Start the integration jobs
    print('Found {0} integration jobs, a maxiumum of {1} was given.'.format(actual_int_jobs, args.integrate))
    print('Starting all jobs.')
    if not args.log:
        print('--- Output may be cluttered. (Try the option -l/--log) ---')
    processes = []
    for i in range(actual_int_jobs):
        # Set up parameters
        parameters = ['runModeList = cms.untracked.string(\'integrate\')']
        parameters.append('integrationList = cms.untracked.string(\'' + str(i) + '\')')
    
        integration_name = template_name + '_integrate_' + str(i) + '.py'
        adjust_pset(args.cmsRunfile, integration_name, parameters)

        cleanupfiles.append(integration_name)
    
        print('Calling\t\'cmsRun ' + integration_name + '\'')
        if args.log:
            print('Writing ouput to log file: ' + integration_name[:-2] + 'log')
            with open(integration_name[:-2] + 'log', 'w') as integration_log:
                processes.append( subprocess.Popen(['cmsRun', integration_name], stdout=integration_log, stderr=subprocess.STDOUT) )
        else:
            processes.append( subprocess.Popen(['cmsRun', integration_name]) )


    # Wait for all processes to finish
    for process in processes:
        process.wait()
    print('--------------------------')
    print('Integration step finished.')
    print('--------------------------')



## Run mode ##

## This part uses the parallelization of the run step provided by
## Herwig. At the moment it is not usable.

##if args.run != 0:
##    parameters = ['runModeList = cms.untracked.string(\'run\')']
##    parameters.append('jobs = cms.untracked.int32(' + str(args.run) + ')')
##
##    run_name = template_name + '_run.py'
##    adjust_pset(args.cmsRunfile, run_name, parameters)
##    cleanupfiles.append(run_name)
##
##    print 'Setting up {0} run jobs.'.format(str(args.run))
##    print 'Calling\n\t\'cmsRun ' + run_name + '\'\nfor the Herwig run step.'.format(str(args.run))
##    process = subprocess.Popen(['cmsRun ' + run_name], shell=True)
##    process.wait()
##    print '------------------'
##    print 'Run step finished.'
##    print '------------------'

## This is the alternative for a paralellized run step. cmsRun is called
## as often as give with the option -r/--run. So the total number of
## generated events is a corresponding multiple of the number of events
## given in the cmsRun file.


if args.stoprun and args.resumerun:
    print('--stoprun AND --resumerun are chosen: run step will be omitted.')
    cleanupandexit(cleanupfiles)

if args.run != 0:
    # Start the run jobs
    print('Setting up {0} runs.'.format(args.run))
    if not args.log:
        print('--- Output may be cluttered. (Try the option -l/--log) ---')
    processes = []
    for i in range(args.run):
        run_name = template_name + '_run_' + str(i) + '.py'

        # Only create new files if this isn't a resumed run
        if not args.resumerun:
            parameters = ['runModeList = cms.untracked.string(\'run\')']
            # Set different seeds
            parameters.append('seed = cms.untracked.int32(' + str(i) + ')')
            adjust_pset(args.cmsRunfile, run_name, parameters)

        # Unless run will be stopped execute the jobs
        if not args.stoprun:
            # Don't mark the files for cleanup if this is a resumed run
            if not args.resumerun:
                cleanupfiles.append(run_name)

            if not os.path.isfile(run_name):
                print('\'' + run_name + '\' not found. It will be skipped.')
                continue

            print('Calling\t\'cmsRun ' + run_name + '\'')
            if args.log:
                print('Writing ouput to log file: ' + run_name[:-2] + 'log')
                with open(run_name[:-2] + 'log', 'w') as run_log:
                    processes.append( subprocess.Popen(['cmsRun', run_name], stdout=run_log, stderr=subprocess.STDOUT) )
            else:
                processes.append( subprocess.Popen(['cmsRun', run_name]) )


    # Wait for all processes to finish
    for process in processes:
        process.wait()
    if args.stoprun:
        print('--stoprun: kept run files and stopped before calling cmsRun')
    print('------------------')
    print('Run step finished.')
    print('------------------')



if not args.keepfiles:
    cleanupandexit(cleanupfiles)
