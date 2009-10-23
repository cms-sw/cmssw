#THIS SCRIPT NEEDS PASSWORD MODIFICATION BEFORE IT CAN RUN
#lookup password for database and replace everywhere it says "PSWD"
echo "Tables before nuking"
pool_list_object_relational_mapping -c "oracle://cms_orcoff_int2r/CMS_COND_PIXEL" -u CMS_COND_PIXEL -p PSWD
echo "Nuking tables..."
sqlplus cms_cond_pixel@cms_orcoff_int2r/PSWD < CondTools/OracleDBA/sql/nuke_tables.sql 
echo "Preparing schema"
#pool_build_object_relational_mapping -f mapping-template-SiPixelGainCalibration.xml -o test.xml -d CondFormatsSiPixelObjects -b -c "sqlite_file:test.db" -u whatever -p wherever
pool_build_object_relational_mapping -f CondTools/OracleDBA/xml/mapping_SiPixelObjects_CMSSW200.xml -d 
CondFormatsSiPixelObjects  -c "oracle://cms_orcoff_int2r/CMS_COND_PIXEL" -u CMS_COND_PIXEL -p PSWD
echo "Upload cabling map"
echo "Offline Gain calib"
echo
echo "List available tags" 
cmscond_list_iov -c oracle://cms_orcoff_int2r/CMS_COND_PIXEL -u cms_cond_pixel -p PSWD
