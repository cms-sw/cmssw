import ROOT as R
import sys

f = R.TFile.Open("dqm_run_only.root")

th1fs = f.Get("TH1Fs")

indices = f.Get("Indices")

expectedIndices = list()
values = list()
nRuns = 10
nLumiPerRun = 1
nHists = 10
startIndex = 0
lastIndex =-1
for i in xrange(0,nRuns):
    for l in xrange(0,nLumiPerRun):
        #we put dummy lumi entries in for each lumi seen
        expectedIndices.append((i+1,l+1,1000,0,0))
    for j in xrange(0,nHists):
        lastIndex +=1
        values.append(("Foo"+str(j), 0, 1.0))
    expectedIndices.append( (i+1,0,3,startIndex,lastIndex) )
    startIndex = lastIndex+1


if nRuns*nHists != th1fs.GetEntries():
    print "wrong number of entries in TH1Fs",th1fs.GetEntries()
    sys.exit(1)

if nRuns+nRuns*nLumiPerRun != indices.GetEntries():
    print "wrong number of entries in Indices", indices.GetEntries()
    sys.exit(1)

for run in xrange(0,nRuns+nRuns*nLumiPerRun):
    indices.GetEntry(run)
    v = (indices.Run,indices.Lumi,indices.Type,indices.FirstIndex,indices.LastIndex)
    if v != expectedIndices[run]:
        print 'ERROR: unexpected value for indices at index :',run
        print ' expected:', expectedIndices[run]
        print ' found:',v
        sys.exit(1)
    if indices.Type !=1000:
        for ihist in xrange(indices.FirstIndex,indices.LastIndex+1):
            index = ihist
            th1fs.GetEntry(ihist)
            v = (th1fs.FullName,th1fs.Flags,th1fs.Value.GetEntries())
            if v != values[index]:
                print 'ERROR: unexpected value for index, runIndex :',index,run
                print ' expected:',values[index]
                print ' found:',v
                sys.exit(1)
print "SUCCEEDED"

