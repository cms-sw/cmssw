#!/usr/bin/env python
#from G.Benelli and Arun Mittal
# 2016 November 17
#Quick script to split a large sqlite file (holding all of our Noise payloads (Run1+Run2) into a set of smaller ones.
import subprocess
#Input IOVs:
#Reference for the use of subprocess Popen to execute a command:
#subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.read()
#Let's prepare a container for the list of IOVs:
IOVs=[]
#for line in subprocess.Popen("conddb --db dump_one_shot_v1.5_447_256581_258507.db list EcalLaserAPDPNRatios_201510078_256581_258507  --limit 1000",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.readlines():
#for line in subprocess.Popen("conddb --noLimit --db dump_one_shot_v1.5_447_232134_256483.db list EcalLaserAPDPNRatios_20151007_232134_256483",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.readlines():
#for line in subprocess.Popen("conddb --noLimit --db offline2016.db list EcalLaserAPDPNRatios_offline2016",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.readlines():
#for line in subprocess.Popen("conddb --noLimit --db dump_one_shot_v1.5_447_274208_279557.db list EcalLaserAPDPNRatios_offline_corr2016_274208_279557",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.readlines():
#for line in subprocess.Popen("conddb --noLimit --db dump_one_shot_v1.5_447_267160_279557_v1_cut0.55.db list EcalLaserAPDPNRatios_offline_corr2016_267160_279557",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.readlines():
#for line in subprocess.Popen("conddb --noLimit --db EcalPedestals_tree.db list EcalPedestals_laser_2016",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.readlines():
for line in subprocess.Popen("conddb --noLimit --db EcalPedestals_timestamp.db list EcalPedestals_timestamp",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.readlines():
  if "EcalCondObjectContainer" in line:
#    IOVs.append(line.split()[0])             #   for run based IOVs
    IOVs.append((line.split()[2].strip(')')).strip('('))             #   for timestamp based IOVs
print IOVs
print "There are %s IOVs!"%len(IOVs)
#Prepare the conddb_import commands template:
#CommandTemplate="conddb_import -f sqlite:SiStripNoise_GR10_v3_offline.db -c sqlite:SiStripNoise_GR10_v3_offline_%s_%s.db -i SiStripNoise_GR10_v4_offline -t SiStripNoise_GR10_v4_offline -b %s -e %s"

#Let's assemble the commands now!
#Let's pick IOVs every 5:
#RelevantIOVs=[(IOV,IOVs[IOVs.index(IOV)+4],IOVs[IOVs.index(IOV)+5]) for IOV in IOVs if IOVs.index(IOV)==0 or ((IOVs.index(IOV)+1)%5==0 and (IOVs.index(IOV)+5)<len(IOVs))]
#RelevantIOVs=[(IOV,IOVs[IOVs.index(IOV)+199],IOVs[IOVs.index(IOV)+200]) for IOV in IOVs if IOVs.index(IOV)==0 or ((IOVs.index(IOV))%200==0 and (IOVs.index(IOV)+200)<len(IOVs))]
#AZ: every 100
RelevantIOVs=[(IOV,IOVs[IOVs.index(IOV)+49],IOVs[IOVs.index(IOV)+50]) for IOV in IOVs if IOVs.index(IOV)==0 or ((IOVs.index(IOV))%50==0 and (IOVs.index(IOV)+50)<len(IOVs))]

RelevantIOVs.append((RelevantIOVs[-1][2],IOVs[-1],IOVs[-1]))

print RelevantIOVs
for i,splitIOVs in enumerate(RelevantIOVs):
  begin=splitIOVs[0]
  end=splitIOVs[1]
  upperLimit=splitIOVs[1]
  print i,begin,end,upperLimit
#  command = "conddb_import -f sqlite:EcalPedestals_tree.db -c sqlite:EcalPedestals_tree_"+str(begin)+"_"+str(end)+".db -i  EcalPedestals_laser_2016 -t EcalPedestals_laser_2016 -b "+str(begin)+" -e "+str(end)
  command = "conddb_import -f sqlite:EcalPedestals_timestamp.db -c sqlite:EcalPedestals_timestamp_"+str(begin)+"_"+str(end)+".db -i  EcalPedestals_timestamp -t EcalPedestals_timestamp -b "+str(begin)+" -e "+str(end)
  print command

  #Now if we want to execute it inside Python uncomment the following two lines:
  STDOUT=subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.read()
  print STDOUT
