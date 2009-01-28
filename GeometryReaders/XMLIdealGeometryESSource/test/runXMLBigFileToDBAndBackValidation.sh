echo start
date
mkdir workarea
cd workarea
mkdir db
mkdir xml
cd db
rm myfile.db
#touch testIdeal.db
rm trXMLFromDB.out
rm twMakeBigFileIdeal.out
rm twMakeDB.out
rm twLoadDBWithXML.out
rm *.log.xml
rm *.log
rm dumpGeoHistoryDBIdealRead
rm dumpSpecsDBIdealRead

echo start write Ideal
date
cmsRun $CMSSW_BASE/src/DetectorDescription/OfflineDBLoader/test/dumpit.py >twMakeBigFileIdeal.out
source $CMSSW_BASE/src/CondTools/Geometry/test/blob_preparation.txt > twMakeDB.out
cmsRun ../../xmlgeometrywriter.py > twLoadDBWithXML.out
echo end write Ideal
date
echo done with all DB writes.

echo start DB read Ideal
date
cmsRun ../../testReadXMLFromDB.py > trXMLFromDB.out
echo done with read DB Ideal
date

mv dumpGeoHistory dumpGeoHistoryDBIdealRead
mv dumpSpecs dumpSpecsDBIdealRead

echo end all DB reads
date
cd ../xml
rm trIdeal.out
rm dumpGeoHistoryXMLIdealRead
rm dumpSpecsXMLIdealRead
rm diffgeomIdeal.out

echo start XML read both
date
cmsRun $CMSSW_BASE/src/DetectorDescription/OfflineDBLoader/test/testreadXMLIdealOnly_cfg.py > trIdeal.out
echo end XML read both
date

mv dumpGeoHistory dumpGeoHistoryXMLIdealRead
mv dumpSpecs dumpSpecsXMLIdealRead
echo done with reading XML

echo doing seds to replace -0 with 0.
date
sed -i '{s/-0.0000/ 0.0000/g}' dumpGeoHistoryXMLIdealRead
cd ../db
sed -i '{s/-0.0000/ 0.0000/g}' dumpGeoHistoryDBIdealRead
cd ../xml

date
echo this will show if there are any inconsistencies when reading the Ideal Geometry
diff dumpGeoHistoryXMLIdealRead ../db/dumpGeoHistoryDBIdealRead > diffgeomIdeal.out

echo ALL DONE!


