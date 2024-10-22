conddb_import -f frontier://FrontierProd/CMS_CONDITIONS -i XMLFILE_CTPPS_Geometry_2016_922V1 -c sqlite:file.db -b 264232 -e 286693 -t XMLFILE_CTPPS_Geometry_101YV3_hlt
conddb_import -f frontier://FrontierProd/CMS_CONDITIONS -i XMLFILE_CTPPS_Geometry_2017_101YV3 -c sqlite:file.db -b 286693 -e 309055 -t XMLFILE_CTPPS_Geometry_101YV3_hlt
conddb_import -f frontier://FrontierProd/CMS_CONDITIONS -i XMLFILE_CTPPS_Geometry_2018_101YV1 -c sqlite:file.db -b 309055 -t XMLFILE_CTPPS_Geometry_101YV3_hlt
conddb_import -f sqlite_file:file.db -c sqlite_file:XMLFILE_CTPPS_HLT_Geometry.db -t XMLFILE_CTPPS_Geometry_101YV3_hlt -i XMLFILE_CTPPS_Geometry_101YV3_hlt
