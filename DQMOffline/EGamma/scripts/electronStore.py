#!/usr/bin/env python3

#========================================================================
#
# This script is used to copy a root file of histograms,
# together with its log files, in a destination directory.
# Log files will be automatically compressed.
#
# Command-line options :
#
#   -f : force the copy, even if the destination file already exists.
#   -r <name> : name of this set of histograms.
#   -m <message> : specific comment about this set of histograms.
#   -a <analyzers> : slash separated list of analyzers.
#   -c <config> : slash separated list of cmsRun configurations.
# 
# Command-line arguments :
#
#   $1 : path of the ROOT file containing the histograms.
#   $... : path of log files.
#   $n : destination directory.
#
#=========================================================================


from __future__ import print_function
import os, sys, datetime, shutil, optparse


#============================================
# display a command and eventually executes
#============================================

def mysystem(command,apply=1):
  print(command)
  if apply==1: return os.system(command)
  elif apply==0:  return 0
  else:
    print('[electronStore.py] UNSUPPORTED ARGUMENT VALUE FOR mysystem(,apply):',apply)
    exit(1)
  

#============================================
# force immediate flushing of stdout
#============================================

class flushfile(object):
  def __init__(self,f):
    self.f = f
  def write(self,x):
    self.f.write(x)
    self.f.flush()

sys.stdout = flushfile(sys.stdout)


#===================================================================
# when called as an independant executable
#===================================================================

if __name__ == "__main__":


  #============================================
  # command-line arguments
  #============================================
    
  parser = optparse.OptionParser()
  parser.add_option("-f", "--force", dest="force", action="store_true", default=False,
    help="force the copy, even if the destination file already exists.")  
  parser.add_option("-r", "--release", dest="release", action="store", default="current",
    help="release name of this set of histograms.")  
  parser.add_option("-m", "--message", dest="message", action="store", default="",
    help="specific comment about this set of histograms")  
  parser.add_option("-a", "--analyzers", dest="analyzers", action="store", default="",
    help="slash separated list of analyzers")  
  parser.add_option("-c", "--configs", dest="configs", action="store", default="",
    help="slash separated list of cmsRun configurations")  
  (options, args) = parser.parse_args()
  
  if len(args)<2:
    print("[electronStore.py] I NEED AT LEAST TWO ARGUMENTS.")
    exit(2)
  store_file = args.pop(0)
  store_dir = args.pop()
  if len(args)>0: store_logs = ' '.join(args)
  else: store_logs = ''
  
  analyzers = options.analyzers.split('/') ;
  configs = options.configs.split('/') ;    
    
      
  #============================================
  # prepare output directory
  #============================================

  if os.path.exists(store_dir)==False:
    os.makedirs(store_dir)


  #============================================
  # check data files
  #============================================

  if os.path.isfile(store_file)==True :
    print("STORE_FILE =",store_file)
  else :
    print("[electronStore.py] FILE DOES NOT EXIST :",store_file)
    exit(3)
    
  if ( store_logs != '' ) :
    print("STORE_LOGS =",store_logs)


  #============================================
  # check if already done
  #============================================
  
  output_file = store_dir+'/'+store_file
  if ( options.force==False and os.path.isfile(output_file)==True ) :
    print("[electronStore.py] ERROR: "+store_file+" ALREADY STORED IN "+store_dir+" !")
    exit(4)


  #============================================
  # copy
  #============================================

  files = [ store_file ]
  for analyzer in analyzers:
    files.append('../plugins/'+analyzer+'.h')
    files.append('../plugins/'+analyzer+'.cc')
  for config in configs:
    files.append(config+'.py')

  mysystem('cp '+' '.join(files)+' '+store_dir)
  
  if ( store_logs != '' ) :
    mysystem('cp '+store_logs+' '+store_dir)
    mysystem('cd '+store_dir+' && gzip -f *.olog')


  #============================================
  # comment
  #============================================

  store_url = store_dir.replace('/afs/cern.ch/cms/','http://cmsdoc.cern.ch/',1)

  links = []
  for analyzer in analyzers:
    links.append('<a href="'+store_url+'/'+analyzer+'.h">'+analyzer+'.h</a>')
    links.append('<a href="'+store_url+'/'+analyzer+'.cc">'+analyzer+'.cc</a>')
  for config in configs:
    links.append('<a href="'+store_url+'/'+config+'.py">'+config+'.py</a>')

  comment_file = open(store_dir+'/'+store_file+'.comment','w')
  print('The <a href="'+store_url+'/'+store_file+'">'+options.release+' histograms</a>', end=' ', file=comment_file)
  if (options.message!=''):
    print(' ('+options.message+')', end=' ', file=comment_file)
  print(' have been prepared with those analyzers and configurations: '+', '.join(links)+'.', end=' ', file=comment_file)
  print(file=comment_file)
  comment_file.close()
  

  #============================================
  # fin
  #============================================
  
  exit(0)
  
  