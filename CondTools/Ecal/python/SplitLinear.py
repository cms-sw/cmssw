#!/usr/bin/env python
#from G.Benelli and Arun Mittal
# 2016 November 17
#Quick script to split a large sqlite file (holding all of our Noise payloads (Run1+Run2) into a set of smaller ones.
import subprocess
IOVs=[]
for line in subprocess.Popen("conddb --noLimit --db Linear.db list EcalLinearCorrections_from2011_offline",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.readlines():
  if "EcalTimeDependentCorrections" in line:
#    IOVs.append(line.split()[0])             #   for run based IOVs
    IOVs.append((line.split()[2].strip(')')).strip('('))             #   for timestamp based IOVs
print IOVs
print "There are %s IOVs!"%len(IOVs)
#Prepare the conddb_import commands template:
#CommandTemplate="conddb_import -f sqlite:SiStripNoise_GR10_v3_offline.db -c sqlite:SiStripNoise_GR10_v3_offline_%s_%s.db -i SiStripNoise_GR10_v4_offline -t SiStripNoise_GR10_v4_offline -b %s -e %s"

#Let's assemble the commands now!
#Let's pick IOVs every 200:
RelevantIOVs=[(IOV,IOVs[IOVs.index(IOV)+199],IOVs[IOVs.index(IOV)+200]) for IOV in IOVs if IOVs.index(IOV)==0 or ((IOVs.index(IOV))%200==0 and (IOVs.index(IOV)+200)<len(IOVs))]

RelevantIOVs.append((RelevantIOVs[-1][2],IOVs[-1],IOVs[-1]))

print RelevantIOVs
for i,splitIOVs in enumerate(RelevantIOVs):
  begin=splitIOVs[0]
  end=splitIOVs[1]
  upperLimit=splitIOVs[1]
  print i,begin,end,upperLimit
  command = "conddb_import -f sqlite:Linear.db -c sqlite:Linear_"+str(begin)+"_"+str(end)+".db -i EcalLinearCorrections_from2011_offline -t EcalLinearCorrections_from2011_offline -b "+str(begin)+" -e "+str(end)
  print command

  #Now if we want to execute it inside Python uncomment the following two lines:
  STDOUT=subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.read()
  print STDOUT
