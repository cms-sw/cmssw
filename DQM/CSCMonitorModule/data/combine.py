#!python

histos={}
missing={}

f=open('CSCDQM_HistoType.txt', 'r')
for h in f:
  histos[h.strip()]=0

f=open('map.txt', 'r')
for h in f:
  a=h.strip().split()
  key=a[0].strip()
  name=a[1].strip()
  if key in histos:
    histos[key]=name
  else:
    missing[key]=name

a=sorted(histos.keys())
for h in a:
  print h,
  if histos[h] != 0:
    print "\t\t\t\t\t",  histos[h]
  else:
    print ""

print "========================= MISSING ============================"

a=missing.keys()
a.sort()
for h in a:
  print h, missing[h]
