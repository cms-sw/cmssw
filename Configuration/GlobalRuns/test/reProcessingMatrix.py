import optparse
import os

usage="--list"
parser = optparse.OptionParser(usage)
parser.add_option("--GT")
parser.add_option("--TLR",default="--customise Configuration/DataProcessing/RecoTLR")
parser.add_option("--options",default="")
parser.add_option("--output",default="RECO,AOD,DQM")
parser.add_option("--rel",default="39X")

(options,args)=parser.parse_args()

com='cmsDriver.py reco -s RAW2DIGI,L1Reco,RECO%s,DQM%s  --data --magField AutoFromDBCurrent --scenario %s --datatier %s --eventcontent %s %s%s --no_exec --python_filename=rereco_%s%s.py --conditions %s '+options.options

#collision config no Alca
os.system(com%('','','pp',options.output,options.output,options.TLR,'.customisePPData','','pp',options.GT))

#cosmics config without Alca
os.system(com%('','','cosmics',options.output,options.output,options.TLR,'.customiseCosmicData','','cosmics',options.GT))

from Configuration.AlCa.autoAlca import autoAlca
for PD in autoAlca:
    recoSpec=''
    scenario='pp'
    customise='.customisePPData'
    output=options.output
    if PD=='Cosmics':
        scenario='cosmics'
        customise='.customiseCosmicData'
        output="RECO,DQM"
    if PD=='HcalNZS':
        recoSpec=':reconstruction_HcalNZS'
        output="RECO,DQM"

    os.system(com%(recoSpec,',ALCA:'+autoAlca[PD],scenario,output,output,options.TLR,customise,PD+'_',scenario,options.GT))


