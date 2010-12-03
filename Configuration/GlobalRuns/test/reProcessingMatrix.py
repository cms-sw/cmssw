import optparse
import os

usage="--list"
parser = optparse.OptionParser(usage)
parser.add_option("--GT")
parser.add_option("--TLR",default="customise Configuration.DataProcessing.reco_TLR_%s")
parser.add_option("--options",default="")
parser.add_option("--output",default="RECO,DQM")
parser.add_option("--rel",default="39X")

(options,args)=parser.parse_args()

com='cmsDriver.py reco -s RAW2DIGI,L1Reco,RECO%s,DQM%s  --data --magField AutoFromDBCurrent --scenario %s --datatier %s --eventcontent %s %s%s --no_exec --python_filename=rereco_%s%s.py --conditions %s '+options.options

#collision config no Alca
os.system(com%('','','pp',options.output,options.output,options.TLR%(options.rel,),'.customisePPData','','pp',options.GT))

#cosmics config without Alca
os.system(com%('','','cosmics',options.output,options.output,options.TLR%(options.rel,),'.customiseCosmicData','','cosmics',options.GT))

from Configuration.PyReleaseValidation.autoAlca import autoAlca
for PD in autoAlca:
    recoSpec=''
    scenario='pp'
    customise='.customisePPData'
    if PD=='Cosmics':
        scenario='cosmics'
        customise='.customiseCosmicData'
    if PD=='HcalNZS':
        recoSpec=':reconstruction_HcalNZS'

    os.system(com%(recoSpec,',ALCA:'+autoAlca[PD],scenario,options.output,options.output,options.TLR,customise,PD+'_',scenario,options.GT))


