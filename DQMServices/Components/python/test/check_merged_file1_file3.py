import ROOT as R
import sys

f = R.TFile.Open("dqm_merged_file1_file3.root")

folder = "TestFolder/"

th1fs = f.Get("TH1Fs")

indices = f.Get("Indices")

# Run,    Lumi,   Type,           FirstIndex,     LastIndex
#    1       1       3(TH1Fs)                0      11
#    1       2       3(TH1Fs)               12      23
#    1       3       3(TH1Fs)               24      35
#    1       4       3(TH1Fs)               36      47
#    1       5       3(TH1Fs)               48      59
#    1       6       3(TH1Fs)               60      71
#    1       7       3(TH1Fs)               72      83
#    1       8       3(TH1Fs)               84      95
#    1       9       3(TH1Fs)               96     107
#    1      10       3(TH1Fs)              108     119
#    1      11       3(TH1Fs)              120     131
#    1      12       3(TH1Fs)              132     143
#    1      13       3(TH1Fs)              144     155
#    1      14       3(TH1Fs)              156     167
#    1      15       3(TH1Fs)              168     179
#    1      16       3(TH1Fs)              180     191
#    1      17       3(TH1Fs)              192     203
#    1      18       3(TH1Fs)              204     215
#    1      19       3(TH1Fs)              216     227
#    1      20       3(TH1Fs)              228     239
#    1       0       3(TH1Fs)              240     251

# expectedIndices is a list that has one entry per raw in the previous
# table, i.e. one entry per index entry! That's why we need to
# integrate over all {RUN,LS}-base histograms (Foo+Bar) *before*
# putting an entry in this list.


# Since we are merging part of different runs, we are indeed producing
# an output file which is the append of the two original files. This
# ease the check since we can perform each one in cascade, pretending
# that each part is identical to each single file. An annoying but
# mandatory multiplication factor of 2 must be added to properly
# compute the index entries, since we are now hosting two "consecutive
# runs".

expectedIndices = list()
values = list()
# First part, related to Run 1 in file 1
nRuns = 1
nHistsFoo = 10
nHistsBar = 2
nLumiPerRun = 10
startIndex = 0
lastIndex =-1
for i in xrange(0, nRuns):
    # LS-based histograms follow
    for l in xrange(0, nLumiPerRun):
        for j in xrange(0, nHistsBar):
            lastIndex += 1
            values.append((folder + "Bar" + str(j) + "_lumi", 0, 55.0))
        for j in xrange(0, nHistsFoo):
            lastIndex += 1
            values.append((folder + "Foo" + str(j) + "_lumi", 0, 1.0))
        expectedIndices.append( (i + 1, l + 1, 3, startIndex, lastIndex) )
        startIndex = lastIndex + 1
    # Run-based histograms follow
    for j in xrange(0, nHistsBar):
        lastIndex += 1
        values.append((folder + "Bar" + str(j), 0, 55.0))
    for j in xrange(0, nHistsFoo):
        lastIndex += 1
        values.append((folder + "Foo" + str(j), 0, 1.0))
    expectedIndices.append( (i + 1, 0, 3, startIndex, lastIndex) )
    startIndex = lastIndex + 1

# Second part, related to Run 2 in file 3
nRuns = 1
nHistsFoo = 10
nHistsBar = 2
nLumiPerRun = 10
for i in xrange(0, nRuns):
    # LS-based histograms follow
    for l in xrange(0, nLumiPerRun):
        for j in xrange(0, nHistsBar):
            lastIndex += 1
            values.append((folder + "Bar" + str(j) + "_lumi", 0, 55.0))
        for j in xrange(0, nHistsFoo):
            lastIndex += 1
            values.append((folder + "Foo" + str(j) + "_lumi", 0, 1.0))
        expectedIndices.append( (i + 2, l + 1, 3, startIndex, lastIndex) )
        startIndex = lastIndex + 1
    # Run-based histograms follow
    for j in xrange(0, nHistsBar):
        lastIndex += 1
        values.append((folder + "Bar" + str(j), 0, 55.0))
    for j in xrange(0, nHistsFoo):
        lastIndex += 1
        values.append((folder + "Foo" + str(j), 0, 1.0))
    expectedIndices.append( (i + 2, 0, 3, startIndex, lastIndex) )
    startIndex = lastIndex + 1


expected = 2*(nRuns*(nHistsFoo + nHistsBar) + nRuns*nLumiPerRun*(nHistsFoo + nHistsBar))
if expected != th1fs.GetEntries():
    print "wrong number of entries in TH1Fs",th1fs.GetEntries(),"expected",expected
    sys.exit(1)

if 2*(nRuns+nRuns*nLumiPerRun) != indices.GetEntries():
    print "wrong number of entries in Indices", indices.GetEntries()
    sys.exit(1)

indexTreeIndex = 0
# First check on Run 1
for run in xrange(0, nRuns):
    for lumi in xrange(0, nLumiPerRun):
        indices.GetEntry(indexTreeIndex)
        v = (indices.Run, indices.Lumi, indices.Type, indices.FirstIndex, indices.LastIndex)
        if v != expectedIndices[indexTreeIndex]:
            print 'ERROR: unexpected value for indices at run, lumi :', indices.Run, indices.Lumi
            print ' expected:', expectedIndices[indexTreeIndex]
            print ' found:', v
            sys.exit(1)
        for ihist in xrange(indices.FirstIndex,indices.LastIndex+1):
            index = ihist
            th1fs.GetEntry(ihist)
            v = (th1fs.FullName,th1fs.Flags,th1fs.Value.GetEntries())
            if v != values[index]:
                print 'ERROR: unexpected value for index, runIndex,lumiIndex :',index,run,lumi
                print ' expected:',values[index]
                print ' found:',v
                sys.exit(1)
        indexTreeIndex +=1
    indices.GetEntry(indexTreeIndex)
    for ihist in xrange(indices.FirstIndex,indices.LastIndex+1):
        index = ihist
        th1fs.GetEntry(ihist)
        v = (th1fs.FullName,th1fs.Flags,th1fs.Value.GetEntries())
        if v != values[index]:
            print 'ERROR: unexpected value for index, runIndex :',index,run
            print ' expected:',values[index]
            print ' found:',v
            sys.exit(1)
    indexTreeIndex +=1

# Second check on Run 2
for run in xrange(0, nRuns):
    for lumi in xrange(0, nLumiPerRun):
        indices.GetEntry(indexTreeIndex)
        v = (indices.Run, indices.Lumi, indices.Type, indices.FirstIndex, indices.LastIndex)
        if v != expectedIndices[indexTreeIndex]:
            print 'ERROR: unexpected value for indices at run, lumi :', indices.Run, indices.Lumi
            print ' expected:', expectedIndices[indexTreeIndex]
            print ' found:', v
            sys.exit(1)
        for ihist in xrange(indices.FirstIndex,indices.LastIndex+1):
            index = ihist
            th1fs.GetEntry(ihist)
            v = (th1fs.FullName,th1fs.Flags,th1fs.Value.GetEntries())
            if v != values[index]:
                print 'ERROR: unexpected value for index, runIndex,lumiIndex :',index,run,lumi
                print ' expected:',values[index]
                print ' found:',v
                sys.exit(1)
        indexTreeIndex +=1
    indices.GetEntry(indexTreeIndex)
    for ihist in xrange(indices.FirstIndex,indices.LastIndex+1):
        index = ihist
        th1fs.GetEntry(ihist)
        v = (th1fs.FullName,th1fs.Flags,th1fs.Value.GetEntries())
        if v != values[index]:
            print 'ERROR: unexpected value for index, runIndex :',index,run
            print ' expected:',values[index]
            print ' found:',v
            sys.exit(1)
    indexTreeIndex +=1

print "SUCCEEDED"

