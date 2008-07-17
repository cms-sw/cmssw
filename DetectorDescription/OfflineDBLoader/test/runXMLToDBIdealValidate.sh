echo start
date
mkdir workarea
cd workarea
mkdir db
mkdir xml
cd db
rm testIdeal.db
touch testIdeal.db
rm trIdeal.out
rm twIdeal.out
rm *.log.xml
rm *.log
rm dumpSpecsIdealWrite
rm dumpGeoHistoryIdealWrite
rm dumpGeoHistoryDBIdealRead
rm dumpSpecsDBIdealRead

echo start write Ideal
date
cmsRun ../../testwriteIdeal_cfg.py >twIdeal.out
echo end write Ideal
date

mv dumpGeoHistory dumpGeoHistoryIdealWrite
mv dumpSpecs dumpSpecsIdealWrite
echo done with all DB writes.

echo start all DB reads.
echo start DB read Ideal
date
cmsRun ../../testreadDBIdealOnly_cfg.py > trIdeal.out
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
cmsRun ../../testreadXMLIdealOnly_cfg.py > trIdeal.out
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


