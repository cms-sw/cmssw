#! /usr/bin/env python

r'''
Relvalreport_v2: a script to run performance tests and produce reports in a automated way.
'''


# Configuration parameters:#############################################################

# Perfreport 3 and 2 coordinates:
PR3_BASE='/afs/cern.ch/user/d/dpiparo/w0/perfreport3installation/'
PR3=PR3_BASE+'/bin/perfreport'# executable
PERFREPORT3_PATH=PR3_BASE+'/share/perfreport' #path to xmls
#PR3_PRODUCER_PLUGIN=PR3_BASE+'/lib/libcmssw_by_producer.so' #plugin for fpage
PR3_PRODUCER_PLUGIN='/afs/cern.ch/user/d/dpiparo/w0/pr3/perfreport/plugins/cmssw_by_producer/libcmssw_by_producer.so'

PR2_BASE='/afs/cern.ch/user/d/dpiparo/w0/perfreport2.1installation/'
PR2='%s' %(PR2_BASE+'/bin/perfreport')# executable
PERFREPORT2_PATH=PR2_BASE+'/share/perfreport' #path to xmls

import os
cmssw_base=os.environ["CMSSW_BASE"]
cmssw_release_base=os.environ["CMSSW_RELEASE_BASE"]
pyrelvallocal=cmssw_base+"/src/Configuration/PyReleaseValidation"
#Set the path depending on the presence of a locally checked out version of PyReleaseValidation
if os.path.exists(pyrelvallocal):
    RELEASE='CMSSW_BASE'
    print "Using LOCAL version of Configuration/PyReleaseValidation instead of the RELEASE version"
elif not os.path.exists(pyrelvallocal):
    RELEASE='CMSSW_RELEASE_BASE'
    
# Valgrind Memcheck Parser coordinates:
VMPARSER='%s/src/Utilities/ReleaseScripts/scripts/valgrindMemcheckParser.pl' %os.environ['CMSSW_RELEASE_BASE']#This has to always point to 'CMSSW_RELEASE_BASE'

# IgProf_Analysis coordinates:
IGPROFANALYS='%s/src/Configuration/PyReleaseValidation/scripts/cmsIgProf_Analysis.py'%os.environ[RELEASE]

# Timereport parser
TIMEREPORTPARSER='%s/src/Configuration/PyReleaseValidation/scripts/cmsTimeReport.pl'%os.environ[RELEASE]

# Simple memory parser
SIMPLEMEMPARSER='%s/src/Configuration/PyReleaseValidation/scripts/cmsSimplememchecker_parser.py' %os.environ[RELEASE]

# Timing Parser
TIMINGPARSER='%s/src/Configuration/PyReleaseValidation/scripts/cmsTiming_parser.py' %os.environ[RELEASE]

# makeSkimDriver
MAKESKIMDRIVERDIR='%s/src/Configuration/EventContent/test' %os.environ[RELEASE]
MAKESKIMDRIVER='%s/makeSkimDriver.py'%MAKESKIMDRIVERDIR

########################################################################################


# Library to include to run valgrind fce
VFCE_LIB='/afs/cern.ch/user/m/moserro/public/vgfcelib' 
PERL5_LIB='/afs/cern.ch/user/d/dpiparo/w0/PERLlibs/5.8.0'



# the profilers that use the stout of the app..
STDOUTPROFILERS=['Memcheck_Valgrind',
                 'Timereport_Parser',
                 'Timing_Parser',
                 'SimpleMem_Parser']
# Profilers list
PROFILERS=['ValgrindFCE',
           'IgProf_perf',
           'IgProf_mem',
           'Edm_Size']+STDOUTPROFILERS
                            

# name of the executable to benchmark. It can be different from cmsRun in future           
EXECUTABLE='cmsRun'

# Command execution and debug switches
EXEC=True
DEBUG=True
 
import time   
import optparse 
import sys


#######################################################################
def red(string):
    return '%s%s%s' %('\033[1;31m',string,'\033[1;0m')    
def green(string):
    return '%s%s%s' %('\033[1;32m',string,'\033[1;0m') 
def yellow(string):
    return '%s%s%s' %('\033[1;33m',string,'\033[1;0m')     
#######################################################################

def clean_name(name):
    '''
    Trivially removes an underscore if present as last char of a string
    '''
    i=-1
    is_dirty=True
    while(is_dirty):
        if name[i]=='_':
            name=name[:-1]
        else:
            return name
        i-=1

#######################################################################

def execute(command):
        '''
        It executes command if the EXEC switch is True. 
        Catches exitcodes different from 0.
        '''
        logger('%s %s ' %(green('[execute]'),command))
        if EXEC:
            exit_code=os.system(command)
            if exit_code!=0:
                logger(red('*** Seems like "%s" encountered problems.' %command))
            return exit_code
        else:
            return 0
            
#######################################################################            
            
def logger(message,level=0):
    '''
    level=0 output, level 1 debug.
    '''                  
    message='%s %s' %(yellow('[RelValreport]'),message)
    
    sys.stdout.flush()
    
    if level==0:
        print message
    if level==1 and DEBUG:
        print message    
    
    sys.stdout.flush()

#######################################################################

class Candles_file:
    '''
    Class to read the trivial ASCCI file containing the candles
    '''
    def __init__(self, filename):
        
        self.commands_profilers_meta_list=[]    
    
        candlesfile=open(filename,'r')
        
        if filename[-3:]=='xml':
            command=''
            profiler=''
            meta=''
            db_meta=''
            reuse=False    
            
            from xml.dom import minidom
            
            # parse the config
            xmldoc = minidom.parse(filename)
            
            # get the candles
            candles_list = xmldoc.getElementsByTagName('candle')
            
            # a list of dictionaries to store the info
            candles_dict_list=[]
            
            for candle in candles_list:
                info_dict={}
                for child in candle.childNodes:# iteration over candle node children
                    if not child.__dict__.has_key('nodeName'):# if just a text node skip!
                        #print 'CONTINUE!'
                        continue
                    # We pick the info from the node
                    tag_name=child.tagName
                    #print 'Manipulating a %s ...'%tag_name
                    data=child.firstChild.data
                    #print 'Found the data: %s !' %data
                    # and we put it in the dictionary
                    info_dict[tag_name]=data
                # to store it in a list
                candles_dict_list.append(info_dict)
            
            # and now process what was parsed
                        
            for candle_dict in candles_dict_list:
                # compulsory params!!
                command=candle_dict['command']
                profiler=candle_dict['profiler']
                meta=candle_dict['meta']
                # other params
                try:
                    db_meta=candle_dict['db_meta']
                except:
                    db_meta=None
                try:
                    reuse=candle_dict['reuse']
                except:
                    reuse=False    
                            
                self.commands_profilers_meta_list.append([command,profiler,meta,reuse,db_meta])
        
        # The file is a plain ASCII
        else:
            for candle in candlesfile.readlines():
                # Some parsing of the file
                if candle[0]!='#' and candle.strip(' \n\t')!='': # if not a comment or an empty line
                    if candle[-1]=='\n': #remove trail \n if it's there
                        candle=candle[:-1] 
                    splitted_candle=candle.split('@@@') #separate
                    
                    # compulsory fields
                    command=splitted_candle[0]
                    profiler=splitted_candle[1].strip(' \t')
                    meta=splitted_candle[2].strip(' \t')        
                    info=[command,profiler,meta]
                    
                    # FIXME: AN .ini or xml config??
                    # do we have something more?
                    len_splitted_candle=len(splitted_candle)
                    reuse=False
                    if len_splitted_candle>3:
                        # is it a reuse statement?
                        if 'reuse' in splitted_candle[3]:
                            reuse=True
                        info.append(reuse)
                    else:
                        info.append(reuse)               
                    
                    # we have one more field or we hadn't a reuse in the last one    
                    if len_splitted_candle>4 or (len_splitted_candle>3 and not reuse):
                        cmssw_scram_version_string=splitted_candle[-1].strip(' \t')
                        info.append(cmssw_scram_version_string)
                    else:
                        info.append(None)
    
                        
                    self.commands_profilers_meta_list.append(info)
                    
    #----------------------------------------------------------------------
        
    def get_commands_profilers_meta_list(self):
        return self.commands_profilers_meta_list
            
#######################################################################

class Profile:
    '''
    Class that represents the procedure of performance report creation
    '''
    def __init__(self,command,profiler,profile_name):
        self.command=command
        self.profile_name=profile_name
        self.profiler=profiler
    
    #------------------------------------------------------------------
    # edit here if more profilers added
    def make_profile(self):
        '''
        Launch the ricght function according to the profiler name.
        '''  
        if self.profiler=='ValgrindFCE':
            return self._profile_valgrindfce()
        elif self.profiler.find('IgProf')!=-1:
            return self._profile_igprof()    
        elif self.profiler.find('Edm_Size')!=-1:
            return self._profile_edmsize()
        elif self.profiler=='Memcheck_Valgrind':
            return self._profile_Memcheck_Valgrind()
        elif self.profiler=='Timereport_Parser':
            return self._profile_Timereport_Parser()
        elif self.profiler=='Timing_Parser':
            return self._profile_Timing_Parser()        
        elif self.profiler=='SimpleMem_Parser':
            return self._profile_SimpleMem_Parser()
        elif self.profiler=='':
            return self._profile_None()
        else:
            raise('No %s profiler found!' %self.profiler)
    #------------------------------------------------------------------
    def _profile_valgrindfce(self):
        '''
        Valgrind profile launcher.
        '''
        # ValgrindFCE needs a special library to run
        os.environ["VALGRIND_LIB"]=VFCE_LIB
        
        profiler_line=''
        valgrind_options= 'time valgrind '+\
                          '--tool=callgrind '+\
                          '--fce=%s ' %(self.profile_name)
        
        # If we are using cmsDriver we should use the prefix switch        
        if EXECUTABLE=='cmsRun' and self.command.find('cmsDriver.py')!=-1:
            profiler_line='%s --prefix "%s"' %(self.command,valgrind_options)
                            
        else:                          
            profiler_line='%s %s' %(valgrind_options,self.command)
                        #'--trace-children=yes '+\
        
        return execute(profiler_line)
    
    #------------------------------------------------------------------
    def _profile_igprof(self): 
        '''
        IgProf profiler launcher.
        '''
        profiler_line=''
        
        igprof_options='igprof -d -t %s ' \
                    %EXECUTABLE # IgProf profile not general only for CMSRUN!
        
        # To handle Igprof memory and performance profiler in one function
        if self.profiler=='IgProf_perf':
            igprof_options+='-pp '
        elif self.profiler=='IgProf_mem':
            igprof_options+='-mp '
        else:
            raise ('Unknown IgProf flavour: %s !'%self.profiler)
        
        igprof_options+='-z -o %s' %(self.profile_name)
        
        # If we are using cmsDriver we should use the prefix switch 
        if EXECUTABLE=='cmsRun' and self.command.find('cmsDriver.py')!=-1:
            profiler_line='%s --prefix "%s"' %(self.command,igprof_options) 
        else:
            profiler_line='%s %s' %(igprof_options, self.command)  
            
        return execute(profiler_line)
    
    #------------------------------------------------------------------
    
    def _profile_edmsize(self):
        '''
        Launch edm size profiler
        '''
        # In this case we replace the name to be clear
        input_rootfile=self.command
        
        # Skim the content if requested!
        if '.' in self.profiler:
            
            clean_profiler_name,options=self.profiler.split('.')
            content,nevts=options.split(',')
            outfilename='%s_%s.root'%(os.path.basename(self.command)[:-6],content)
            oldpypath=os.environ['PYTHONPATH']
            os.environ['PYTHONPATH']+=':%s' %MAKESKIMDRIVERDIR
            execute('%s -i %s -o %s --outputcommands %s -n %s' %(MAKESKIMDRIVER,
                                                                 self.command,
                                                                 outfilename,
                                                                 content,
                                                                 nevts)) 
            os.environ['PYTHONPATH']=oldpypath
            #execute('rm %s' %outfilename)
            self.command=outfilename
            self.profiler=clean_profiler_name
                                                                                                        
        
        profiler_line='edmEventSize -o %s -d %s'\
                            %(self.profile_name,self.command)
        
        return execute(profiler_line)
   
   #------------------------------------------------------------------
   
    def _profile_Memcheck_Valgrind(self):
        '''
        Launch Valgrind Memcheck profiler
        '''
        
        profiler_line='valgrind --tool=memcheck '+\
                               '--leak-check=yes '+\
                               ' --show-reachable=yes '+\
                               '--num-callers=20 '+\
                               '--track-fds=yes '+\
                               '%s 2>&1 |tee %s' %(self.command,self.profile_name)
        
        return execute(profiler_line)
        
    #-------------------------------------------------------------------
    
    def _profile_Timereport_Parser(self):
        return self._save_output()
        
    #-------------------------------------------------------------------
    
    def _profile_SimpleMem_Parser(self):
        return self._save_output()

    #-------------------------------------------------------------------
    
    def _profile_Timing_Parser(self):
        return self._save_output()     
    
    #-------------------------------------------------------------------
        
    def _save_output(self):
        '''
        Save the output of cmsRun on a file!
        '''               
#         # a first maquillage about the profilename:
#         if self.profile_name[-4:]!='.log':
#             self.profile_name+='.log'
        
        profiler_line='%s  2>&1 |tee %s' %(self.command,self.profile_name)
        return execute(profiler_line)    

    #-------------------------------------------------------------------                    
    
    def _profile_None(self):
        '''
        Just Run the command!
        '''
        execute(self.command)
    
    #-------------------------------------------------------------------
    
    def make_report(self,
                    fill_db=False,
                    db_name=None,
                    tmp_dir=None,
                    outdir=None,
                    IgProf_option=None,
                    metastring=None):
        '''
        Make a performance report with perfreport 3 or 2. PR2 will be no longer supported in future.
        '''

        
        if outdir==None or outdir==self.profile_name:
            outdir=self.profile_name+'_outdir'            
                  
        if not os.path.exists(outdir) and not fill_db:
            execute('mkdir %s' %outdir)
        
        if fill_db:
            db_option='-a'
            if not os.path.exists(db_name):
                db_option='-A'
        
        # temp in the local dir for PR
        tmp_switch=''    
        if tmp_dir!='':
            execute('mkdir %s' %tmp_dir)
            tmp_switch=' -t %s' %tmp_dir
        
        # Profiler is ValgrindFCE:
        if self.profiler=='ValgrindFCE':
            perfreport_command=''
            # Switch for filling the db
            if not fill_db:
                os.environ["PERFREPORT_PATH"]='%s/' %PERFREPORT2_PATH
                perfreport_command='%s %s -ff -i %s -o %s' %(PR2,
                                                             tmp_switch,
                                                             self.profile_name,
                                                             outdir)
            else:
                os.environ["PERFREPORT_PATH"]='%s/' %PERFREPORT3_PATH
                perfreport_command='%s %s -n5000 -u%s  -ff  -m \'scram_cmssw_version_string,%s\' -i %s %s -o %s' \
                                                    %(PR3,
                                                      tmp_switch,
                                                      PR3_PRODUCER_PLUGIN,
                                                      metastring,
                                                      self.profile_name,
                                                      db_option,
                                                      db_name)
            execute(perfreport_command)
        
        #####################################################################            
            
        # Profiler is IgProf:
        if self.profiler.find('IgProf')!=-1:
            if IgProf_option!='ANALYSE':
                uncompressed_profile_name=self.profile_name[:-3]+'_uncompressed'
                execute('gzip -d -c %s > %s' %(self.profile_name,uncompressed_profile_name))
                perfreport_command=''
                # Switch for filling the db
                if not fill_db:
                    os.environ["PERFREPORT_PATH"]='%s/' %PERFREPORT2_PATH
                    perfreport_command='%s %s -fi -y %s -i %s -o %s' \
                                    %(PR2,
                                      tmp_switch,
                                      IgProf_option,
                                      uncompressed_profile_name,
                                      outdir)
                else:
                    os.environ["PERFREPORT_PATH"]='%s/' %PERFREPORT3_PATH
                    perfreport_command='%s %s -n5000 -u%s  -fi -m \'scram_cmssw_version_string,%s\' -y %s -i %s %s -o %s' \
                                    %(PR3,
                                      tmp_switch,
                                      PR3_PRODUCER_PLUGIN,
                                      metastring,
                                      IgProf_option,
                                      uncompressed_profile_name,
                                      db_option,db_name)            
                
                execute(perfreport_command)
                execute('rm  %s' %uncompressed_profile_name)
                
            else: #We use IgProf Analisys
                execute('%s -o%s -i%s' %(IGPROFANALYS,outdir,self.profile_name))
                

        #####################################################################                     
            
        # Profiler is EdmSize:        
        if 'Edm_Size' in self.profiler:
            perfreport_command=''
            if not fill_db:
                os.environ["PERFREPORT_PATH"]='%s/' \
                                            %PERFREPORT2_PATH
                perfreport_command='%s %s -fe -i %s -o %s' \
                                            %(PR2,
                                              tmp_switch,
                                              self.profile_name,
                                              outdir)
            else:
                os.environ["PERFREPORT_PATH"]='%s/' \
                                            %PERFREPORT3_PATH
                perfreport_command='%s %s -n5000 -u%s -fe -i %s -a -o %s' \
                                            %(PR3,
                                              tmp_switch,
                                              PR3_PRODUCER_PLUGIN,
                                              self.profile_name,
                                              db_name)             

            execute(perfreport_command)    
        
        if tmp_dir!='':
            execute('rm -r %s' %tmp_dir)

        #####################################################################    
                            
        # Profiler is Valgrind Memcheck
        if self.profiler=='Memcheck_Valgrind':
            # Three pages will be produced:
            os.environ['PERL5LIB']=PERL5_LIB
            report_coordinates=(VMPARSER,self.profile_name,outdir)
            report_commands=('%s --preset +prod,-prod1 %s > %s/edproduce.html'\
                                %report_coordinates,
                             '%s --preset +prod1 %s > %s/esproduce.html'\
                                %report_coordinates,
                             '%s -t beginJob %s > %s/beginjob.html'\
                                %report_coordinates)
            for command in report_commands:
                execute(command)      
        
        #####################################################################                
                
        # Profiler is TimeReport parser
        
        if self.profiler=='Timereport_Parser':
            execute('%s %s %s' %(TIMEREPORTPARSER,self.profile_name,outdir))

        #####################################################################
        
        # Profiler is Timing Parser            
        
        if self.profiler=='Timing_Parser':
            execute('%s -i %s -o %s' %(TIMINGPARSER,self.profile_name,outdir))
        
                    
        #####################################################################
        
        # Profiler is Simple memory parser
        
        if self.profiler=='SimpleMem_Parser':
            execute('%s -i %s -o %s' %(SIMPLEMEMPARSER,self.profile_name,outdir))
            
        #####################################################################
        
        # no profiler
            
        if self.profiler=='':
            pass                    
                                                                
#############################################################################################

def principal(options):
    '''
    Here the objects of the Profile class are istantiated.
    '''
    
    # Build a list of commands for programs to benchmark.
    # It will be only one if -c option is selected
    commands_profilers_meta_list=[]
    
    # We have only one
    if options.infile=='':
        logger('Single command found...')
        commands_profilers_meta_list.append([options.command,'','',False,''])
    
    # We have more: we parse the list of candles    
    else:
        logger('List of commands found. Processing %s ...' %options.infile)
        
        # an objects that represents the cndles file:
        candles_file = Candles_file(options.infile)
        
        commands_profilers_meta_list=candles_file.get_commands_profilers_meta_list()
       
        
    logger('Iterating through commands of executables to profile ...')
    
    # Cycle through the commands
    len_commands_profilers_meta_list=len(commands_profilers_meta_list)    
    
    commands_counter=1
    precedent_profile_name=''
    precedent_reuseprofile=False
    for command,profiler_opt,meta,reuseprofile,db_metastring in commands_profilers_meta_list:
                  
        exit_code=0
        
        logger('Processing command %d/%d' \
                    %(commands_counter,len_commands_profilers_meta_list))
        logger('Process started on %s' %time.asctime())
        
        # for multiple directories and outputs let's put the meta
        # just before the output profile and the outputdir
        profile_name=''
        profiler=''
        reportdir=options.output
        IgProf_counter=options.IgProf_counter
           
        
        if options.infile!='': # we have a list of commands

            reportdir='%s_%s' %(meta,options.output)
            reportdir=clean_name(reportdir)                        
         
            profile_name=clean_name('%s_%s'%(meta,options.profile_name))
                    
            # profiler is igprof: we need to disentangle the profiler and the counter
            if profiler_opt.find('.')!=-1 and \
               profiler_opt.find('IgProf')!=-1:
                profiler_opt_split=profiler_opt.split('.')
                profiler,IgProf_counter=profiler_opt_split
                if profile_name[-3:]!='.gz':
                    profile_name+='.gz'
                    
            elif profiler_opt.find('MEM_TOTAL')!=-1 or\
                 profiler_opt.find('MEM_LIVE')!=-1 or\
                 profiler_opt.find('MEM_PEAK')!=-1: 
                profiler='IgProf_mem'
                IgProf_counter=profiler_opt


                if profile_name[-3:]!='.gz':
                    profile_name+='.gz'
            
            # Profiler is Timereport_Parser
            elif profiler_opt in STDOUTPROFILERS:
                # a first maquillage about the profilename:
                if profile_name[:-4]!='.log':
                    profile_name+='.log'
                profiler=profiler_opt
                        
            # profiler is not igprof
            else:
                profiler=profiler_opt
            
            if precedent_reuseprofile:
                profile_name=precedent_profile_name
            if reuseprofile:
                precedent_profile_name=profile_name 

                                
                        
        else: # we have a single command: easy job!
            profile_name=options.profile_name
            reportdir=options.output
            profiler=options.profiler

        
        
        # istantiate a Profile object    
        if precedent_profile_name!='':
            if os.path.exists(precedent_profile_name):
                logger('Reusing precedent profile: %s ...' %precedent_profile_name)
                logger('Copying the old profile to the new name %s ...' %profile_name)
                execute('cp %s %s' %(precedent_profile_name, profile_name))
                
        performance_profile=Profile(command,
                                    profiler,
                                    profile_name)   

        # make profile if needed
        if options.profile:                                         
            if reuseprofile:
                logger('Saving profile name to reuse it ...')
                precedent_profile_name=profile_name
            else:
                precedent_profile_name=''                                
            
            if not precedent_reuseprofile:
                logger('Creating profile for command %d using %s ...' \
                                                %(commands_counter,profiler))     
                exit_code=performance_profile.make_profile()
            
            
        
        # make report if needed   
        if options.report:
            if exit_code!=0:
                logger('Halting report creation procedure: unexpected exit code %s from %s ...' \
                                            %(exit_code,profiler))
            else:   
                logger('Creating report for command %d using %s ...' \
                                                %(commands_counter,profiler))     
                                               
                # Write into the db instead of producing html if this is the case:
                if options.db:
                    performance_profile.make_report(fill_db=True,
                                                    db_name=options.output,
                                                    metastring=db_metastring,
                                                    tmp_dir=options.pr_temp,
                                                    IgProf_option=IgProf_counter)
                else:
                    performance_profile.make_report(outdir=reportdir,
                                                    tmp_dir=options.pr_temp,
                                                    IgProf_option=IgProf_counter)
        commands_counter+=1                                                
        precedent_reuseprofile=reuseprofile
        if not precedent_reuseprofile:
            precedent_profile_name=''
        
        logger('Process ended on %s\n' %time.asctime())
    
    logger('Procedure finished on %s' %time.asctime())      

###################################################################################################    
        
if __name__=="__main__":

    usage='\n'+\
          '----------------------------------------------------------------------------\n'+\
          ' RelValreport: a tool for automation of benchmarking and report generation. \n'+\
          '----------------------------------------------------------------------------\n\n'+\
          'relvalreport.py <options>\n'+\
          'relvalreport.py -i candles_150.txt -R -P -n 150.out -o 150_report\n'+\
          ' - Executes the candles contained in the file candles_150.txt, create\n'+\
          '   profiles, specified by -n, and reports, specified by -o.\n\n'+\
          'Candles file grammar:\n'+\
          'A candle is specified by the syntax:\n'+\
          'executable_name @@@ profiler_name @@@ meta\n'+\
          ' - executable_name: the name of the executable to benchmark.\n'+\
          ' - profiler_name: the name of the profiler to use. The available are: %s.\n' %str(PROFILERS)+\
          '   In case you want to use IgProf_mem or IgProf_perf, the counter (MEM_TOTAL,PERF_TICKS...)\n'+\
          '   must be added with a ".": IgProf_mem.MEM_TOTAL.\n'+\
          ' - meta: metastring that is used to change the name of the names specified in the command line\n'+\
          '   in case of batch execution.'+\
          'An example of candle file:\n\n'+\
          '  ># My list of candles:\n'+\
          '  >\n'+\
          '  >cmsDriver.py MU- -sSIM  -e10_20 @@@ IgProf_perf.PERF_TICKS @@@ QCD_sim_IgProfperf\n'+\
          '  >cmsDriver.py MU- -sRECO -e10_20 @@@ ValgrindFCE            @@@ QCD_reco_Valgrind\n'+\
          '  >cmsRun mycfg.cfg                @@@ IgProf_mem.MEM_TOTAL   @@@ Mycfg\n'
             
             

    parser = optparse.OptionParser(usage)

    parser.add_option('-p', '--profiler',
                      help='Profilers are: %s' %str(PROFILERS) ,
                      default='',
                      dest='profiler')
                                            
    parser.add_option('-c', '--command',
                      help='Command to profile. If specified the infile is ignored.' ,
                      default='',
                      dest='command') 
                      
    parser.add_option('-t',
                      help='The temp directory to store the PR service files. Default is PR_TEMP Ignored if PR is not used.',
                      default='',
                      dest='pr_temp')
    
    #Flags                      
                      
    parser.add_option('--db',
                      help='EXPERIMENTAL: Write results on the db.',
                      action='store_true',
                      default=False,
                      dest='db')               

    parser.add_option('-R','--Report',
                      help='Create a static html report. If db switch is on this is ignored.',
                      action='store_true',
                      default=False,
                      dest='report')                      
    
    parser.add_option('-P','--Profile',
                      help='Create a profile for the selected profiler.',
                      action='store_true',
                      default=False,
                      dest='profile')                        
    
    # Output location for profile and report                      
    
    parser.add_option('-n', '--profile_name',
                      help='Profile name' ,
                      default='',
                      dest='profile_name') 
                                                                                                                            
    parser.add_option('-o', '--output',
                      help='Outdir for the html report or db filename.' ,
                      default='',
                      dest='output')
                      
    #Batch functionality                       
                                               
    parser.add_option('-i', '--infile',
                      help='Name of the ASCII file containing the commands to profile.' ,
                      default='',
                      dest='infile')    
    
    # ig prof options

    parser.add_option('-y', 
                      help='Specify the IgProf counter or the CMSSW. '+\
                           'If a profiler different from '+\
                           'IgProf is selected this is ignored.' ,
                      default='PERF_TICKS',
                      dest='IgProf_counter')                        
                      
    parser.add_option('--executable',
                      help='Specify executable to monitor if different from cmsRun. '+\
                           'Only valid for IgProf.',
                      default='',
                      dest='executable')                               
              
    # Debug options
    parser.add_option('--noexec',
                      help='Do not exec commands, just display them!',
                      action='store_true',
                      default=False,
                      dest='noexec')   
                                        
    (options,args) = parser.parse_args()
    
    # FAULT CONTROLS
    if options.infile=='' and options.command=='' and not (options.report and not options.profile):
        raise('Specify at least one command to profile!')
    if options.profile_name=='' and options.infile=='':
        raise('Specify a profile name!')
    if not options.db and options.output=='' and options.infile=='':
        raise('Specify a db name or an output dir for the static report!')
    
    if not options.profile:
        if not os.path.exists(options.profile_name) and options.infile=='':
            raise('Profile %s does not exist!' %options.profile_name)
        logger("WARNING: No profile will be generated. An existing one will be processed!")
    
    if options.command!='' and options.infile!='':
        raise('-c and -i options cannot coexist!')    
    
    if options.profiler=='Memcheck_Valgrind' and not os.path.exists(VMPARSER):
        raise('Couldn\'t find Valgrind Memcheck Parser Script! Please install it from Utilities/ReleaseScripts.')
            
    if options.executable!='':
        globals()['EXECUTABLE']=options.executable
    
    if options.noexec:
        globals()['EXEC']=False
        
    logger('Procedure started on %s' %time.asctime())                               
    
    if options.infile == '':
        logger('Script options:')
        for key,val in options.__dict__.items():
            if val!='':
                logger ('\t\t|- %s = %s' %(key, str(val)))
                logger ('\t\t|')
    
    principal(options)
                
