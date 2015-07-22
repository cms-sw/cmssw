#!/usr/bin/env python

#========================================================================
#
# This script is used to generate a web page which superpose two
# sets of similar histograms.
#
# Command-line options :
#
#   -c <configuration> : description of the histograms to be displayed and how.
#   -t <title> : general title of the page.
#   -r <name> : short name of the red histograms.
#   -b <name> : short name of the blue histograms.
# 
# Command-line arguments :
#
#   $1 : path of the ROOT file containing the red histograms.
#   $2 : path of the ROOT file containing the blue histograms.
#   $3 : destination directory.
#
#=========================================================================


import os, sys, datetime, shutil, optparse


#============================================
# display a command and eventually executes
#============================================

def mysystem(command,apply=1):
  print command
  if apply==1: return os.system(command)
  elif apply==0:  return 0
  else:
    print '[electronStore.py] UNSUPPORTED ARGUMENT VALUE FOR mysystem(,apply):',apply
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
  parser.add_option("-c", "--cfg", dest="config", action="store", default="electronCompare.txt",
    help="the configuration file describe which histogram must be displayed and how")  
  parser.add_option("-t", "--title", dest="title", action="store", default="",
    help="the title of the page")  
  parser.add_option("-r", "--red-name", dest="red", action="store", default="",
    help="short name of the red histograms")  
  parser.add_option("-b", "--blue-name", dest="blue", action="store", default="",
    help="short name of the blue histograms")  
  (options, args) = parser.parse_args()
#  print "options : ",options
  
  if len(args)<2:
    print "[electronStore.py] I NEED AT LEAST TWO ARGUMENTS."
    exit(2)
    
  red_file = args.pop(0)
  web_dir = args.pop()
  web_url = web_dir.replace('/afs/cern.ch/cms/','http://cmsdoc.cern.ch/',1)
  if len(args)>0 :
    blue_file = args.pop(0)
  else :
    blue_file = ''
      
      
  #===================================================
  # prepare output directories and check input files
  #===================================================

  # destination dir
  print 'WEB DIR =',web_dir
  if os.path.exists(web_dir+'/gifs')==False:
    os.makedirs(web_dir+'/gifs')
    
  # red file
  red_base = os.path.basename(red_file)
  if os.path.isfile(red_file)==True :
    print 'RED FILE =',red_file
    if os.path.isfile(red_base)==True and os.path.getmtime(red_base)>os.path.getmtime(red_file) :
      print '[electronCompare.py] did you forget to store '+red_base+' ?'
  else :
    print "[electronCompare.py] FILE NOT FOUND :",red_file
    if os.path.isfile(red_base)==True :
      print '[electronCompare.py] did you forget to store '+red_base+' ?'
    exit(3)
 
  # blue file
  if blue_file!='' :
    if os.path.isfile(blue_file)==True :
      print 'BLUE FILE =',blue_file
    else :
      print '[electronCompare.py] file not found :',blue_file
      blue_file = ''
  else :
    print "[electronCompare.py] no blue histograms to compare with."
         
      
  #===================================================
  # improved default options
  #===================================================

  (red_head,red_tail) = os.path.split(red_file)
  red_long_name = os.path.basename(red_head)+'/'+red_tail
  (blue_head,blue_tail) = os.path.split(blue_file)
  blue_long_name = os.path.basename(blue_head)+'/'+blue_tail
  if options.red=='' :
    options.red = red_long_name
  if options.blue=='' :
    options.blue = blue_long_name
  if options.title=='' :
    options.title = red_long_name+' vs '+blue_long_name
         
  (red_hd, red_release) = os.path.split(red_head)
  (blue_hd, blue_release) = os.path.split(blue_head)

  #============================================
  # final commands
  #============================================

  mysystem('cp -f electronCompare.C '+options.config+' '+web_dir)
  
  os.environ['CMP_DIR'] = web_dir
  os.environ['CMP_URL'] = web_url
  os.environ['CMP_TITLE'] = options.title
  os.environ['CMP_RED_FILE'] = red_file
  os.environ['CMP_BLUE_FILE'] = blue_file
  os.environ['CMP_RED_NAME'] = options.red
  os.environ['CMP_BLUE_NAME'] = options.blue
  os.environ['CMP_RED_COMMENT'] = red_file+'.comment'
  os.environ['CMP_BLUE_COMMENT'] = blue_file+'.comment'
  os.environ['CMP_CONFIG'] = options.config
  os.environ['CMP_RED_RELEASE'] = red_release
  os.environ['CMP_BLUE_RELEASE'] = blue_release
  
  mysystem('root -b -l -q electronCompare.C')
  
  print "You can access the files here:",web_dir
  print "You can browse your validation plots here:",web_url+'/'
