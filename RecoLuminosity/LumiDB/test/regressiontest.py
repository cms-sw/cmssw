import commands,sys

datatag='--datatag v9'
normtag='--normtag HFV2c'
#refrunnum='203894'
refrunnum='198271'
reffillnum='2797'
refbegtime='"07/02/12 9:15:00"'
refendtime='"07/08/12 03:03:12"'
refminrun='198049'
refmaxrun='198486'
refminfill='2797'
refmaxfill='2816'

refjsonfile='/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions12/8TeV/Reprocessing/Cert_198022-198523_8TeV_24Aug2012ReReco_Collisions12_JSON.txt'
refhltpath='HLT_IsoMu24_eta*'
#refl1bit='L1_Mu7er_ETM*'
refl1bit='L1_SingleMu*'

runnumfilter='-r '+refrunnum

fillnumfilter='-f '+reffillnum

filefilter='-i '+refjsonfile

hltpathfilter='--hltpath '+refhltpath

rangetimefilter='--begin '+refbegtime+' --end '+refendtime
rangerunfilter='--begin '+refminrun+' --end '+refmaxrun
rangefillfilter='--begin '+refminfill+' --end '+refmaxfill

opentimefilter='--begin '+refbegtime
openrunfilter='--begin '+refminrun
openfillfilter='--begin '+refminfill

actionobj=['overview','delivered','lumibyls','recorded','lumibylsXing']
#======================================================================
'''
objective:
  action overview
  runnumfilter
  explicit --datatag vs default 
  explicit --normtag vs default
'''

cmmdbase='lumiCalc2.py --without-checkforupdate --headerfile head.txt'
actionobj=['overview','delivered','lumibyls']
for action in actionobj:
    cmmd=[]
    cmmd.append(cmmdbase)
    cmmd.append(action)
    cmmd.append(runnumfilter)
    cmmdStr=' '.join(cmmd)
    print cmmdStr
    (status,output1) = commands.getstatusoutput(cmmdStr)

    cmmd.append(datatag)
    cmmd.append(normtag)
    cmmdStr=' '.join(cmmd)
    print cmmdStr
    (status,output2) = commands.getstatusoutput(cmmdStr)

    if output1[output1.index('='):]==output2[output2.index('='):]:
        print '    OK'

actionobj=['recorded','lumibyls']
for action in actionobj:
    cmmd=[]
    cmmd.append(cmmdbase)
    cmmd.append(action)
    cmmd.append(runnumfilter)
    cmmd.append(hltpathfilter)
    cmmdStr=' '.join(cmmd)
    print cmmdStr
    (status,output1) = commands.getstatusoutput(cmmdStr)
    
    cmmd.append(datatag)
    cmmd.append(normtag)
    cmmdStr=' '.join(cmmd)
    print cmmdStr
    (status,output2) = commands.getstatusoutput(cmmdStr)

    if output1[output1.index('='):]==output2[output2.index('='):]:
        print '    OK'
    else:
        sys.exit(-1)

actionobj=['overview','lumibyls']
print '========check overlapping -r -i========'
for action in actionobj:
    cmmd=[]
    cmmd.append(cmmdbase)
    cmmd.append(action)
    cmmd.append(runnumfilter)
    cmmd.append(filefilter)
    cmmdStr=' '.join(cmmd)
    print cmmdStr
    (status,output) = commands.getstatusoutput(cmmdStr)
    print output
    if action=='overview':
        if '198049:' not in output:
            print 'OK'
        else:
            sys.exit(-1)
    if action=='lumibyls':
        if '92:92' not in output:
            print 'OK'
        else:
            sys.exit(-1)
actionobj=['delivered','lumibyls']    
print '========check overlapping -f -i========='
for action in actionobj:
    cmmd=[]
    cmmd.append(cmmdbase)
    cmmd.append(action)
    cmmd.append(fillnumfilter)
    cmmd.append(filefilter)
    cmmdStr=' '.join(cmmd)
    print cmmdStr
    (status,output) = commands.getstatusoutput(cmmdStr)
    print output

#actionobj=['delivered','lumibyls']    
#print 'check overlapping -f -i --begin time --end time'
#for action in actionobj:
#    cmmd=[]
#    cmmd.append(cmmdbase)
#    cmmd.append(action)
#    cmmd.append(rangetimefilter)
#    cmmd.append(filefilter)
#    cmmdStr=' '.join(cmmd)
#    print cmmdStr
#    (status,output) = commands.getstatusoutput(cmmdStr)
#    print output
