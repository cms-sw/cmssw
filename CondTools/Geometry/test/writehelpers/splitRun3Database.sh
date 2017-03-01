#!/bin/sh
conddb_import -f sqlite_file:myfile.db -c sqlite_file:GeometryFileExtended2023.db -t XMLFILE_Geometry_TagXX_Extended2023_mc -i XMLFILE_Geometry_TagXX_Extended2023_mc
conddb_import -f sqlite_file:myfile.db -c sqlite_file:GeometryFileIdeal2023.db -t XMLFILE_Geometry_TagXX_Ideal2023_mc -i XMLFILE_Geometry_TagXX_Ideal2023_mc
conddb_import -f sqlite_file:myfile.db -c sqlite_file:TKRECO_Geometry.db -t TKRECO_Geometry2023_TagXX -i TKRECO_Geometry2023_TagXX
conddb_import -f sqlite_file:myfile.db -c sqlite_file:TKExtra_Geometry.db -t TKExtra_Geometry2023_TagXX -i TKExtra_Geometry2023_TagXX
conddb_import -f sqlite_file:myfile.db -c sqlite_file:TKParameters_Geometry.db -t TKParameters_Geometry2023_TagXX -i TKParameters_Geometry2023_TagXX
conddb_import -f sqlite_file:myfile.db -c sqlite_file:EBRECO_Geometry.db -t EBRECO_Geometry2023_TagXX -i EBRECO_Geometry2023_TagXX
conddb_import -f sqlite_file:myfile.db -c sqlite_file:EERECO_Geometry.db -t EERECO_Geometry2023_TagXX -i EERECO_Geometry2023_TagXX
conddb_import -f sqlite_file:myfile.db -c sqlite_file:EPRECO_Geometry.db -t EPRECO_Geometry2023_TagXX -i EPRECO_Geometry2023_TagXX
conddb_import -f sqlite_file:myfile.db -c sqlite_file:HCALRECO_Geometry.db -t HCALRECO_Geometry2023_TagXX -i HCALRECO_Geometry2023_TagXX
conddb_import -f sqlite_file:myfile.db -c sqlite_file:HCALParameters_Geometry.db -t HCALParameters_Geometry2023_TagXX -i HCALParameters_Geometry2023_TagXX
conddb_import -f sqlite_file:myfile.db -c sqlite_file:CTRECO_Geometry.db -t CTRECO_Geometry2023_TagXX -i CTRECO_Geometry2023_TagXX
conddb_import -f sqlite_file:myfile.db -c sqlite_file:ZDCRECO_Geometry.db -t ZDCRECO_Geometry2023_TagXX -i ZDCRECO_Geometry2023_TagXX
conddb_import -f sqlite_file:myfile.db -c sqlite_file:CASTORRECO_Geometry.db -t CASTORRECO_Geometry2023_TagXX -i CASTORRECO_Geometry2023_TagXX
conddb_import -f sqlite_file:myfile.db -c sqlite_file:CSCRECO_Geometry.db -t CSCRECO_Geometry2023_TagXX -i CSCRECO_Geometry2023_TagXX
conddb_import -f sqlite_file:myfile.db -c sqlite_file:CSCRECODIGI_Geometry.db -t CSCRECODIGI_Geometry2023_TagXX -i CSCRECODIGI_Geometry2023_TagXX
conddb_import -f sqlite_file:myfile.db -c sqlite_file:DTRECO_Geometry.db -t DTRECO_Geometry2023_TagXX -i DTRECO_Geometry2023_TagXX
conddb_import -f sqlite_file:myfile.db -c sqlite_file:RPCRECO_Geometry.db -t RPCRECO_Geometry2023_TagXX -i RPCRECO_Geometry2023_TagXX
