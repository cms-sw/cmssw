import ROOT
import sys
db_sid = 'CMS_OMDS_LB'
db_usr = 'CMS_ECAL_LASER_COND'
db_pwd = sys.argv[1]

runMin = 309000
runMax = 999999

econn = ROOT.EcalCondDBInterface( db_sid, db_usr, db_pwd )
my_locdef  = ROOT.LocationDef()
my_locdef.setLocation("P5_Co")
my_rundef  = ROOT.RunTypeDef()
my_rundef.setRunType("PHYSICS")

runtag = ROOT.RunTag()
runtag.setLocationDef(my_locdef)
runtag.setRunTypeDef(my_rundef)
runtag.setGeneralTag("GLOBAL")
runlist = econn.fetchNonEmptyGlobalRunList( runtag, runMin, runMax ).getRuns()
runs =  [ runlist[i].getRunNumber() for i in range( runlist.size() ) ]
print(runs)
