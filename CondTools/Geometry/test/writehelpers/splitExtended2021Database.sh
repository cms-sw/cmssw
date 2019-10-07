#!/bin/sh

conddb --db myfile.db XMLFILE_Geometry_TagXX_Extended2021_mc XMLFILE_Geometry_TagXX_Extended2021_mc --destdb GeometryFileExtended2021.db
conddb --db myfile.db TKRECO_Geometry_TagXX TKRECO_Geometry_TagXX --destdb TKRECO_Geometry.db
conddb --db myfile.db TKExtra_Geometry_TagXX TKExtra_Geometry_TagXX --destdb TKExtra_Geometry.db
conddb --db myfile.db TKParameters_Geometry_TagXX TKParameters_Geometry_TagXX --destdb TKParameters_Geometry.db
conddb --db myfile.db EBRECO_Geometry_TagXX EBRECO_Geometry_TagXX --destdb EBRECO_Geometry.db
conddb --db myfile.db EERECO_Geometry_TagXX EERECO_Geometry_TagXX --destdb EERECO_Geometry.db
conddb --db myfile.db EPRECO_Geometry_TagXX EPRECO_Geometry_TagXX --destdb EPRECO_Geometry.db
conddb --db myfile.db HCALRECO_Geometry_TagXX HCALRECO_Geometry_TagXX --destdb HCALRECO_Geometry.db
conddb --db myfile.db HCALParameters_Geometry_TagXX HCALParameters_Geometry_TagXX --destdb HCALParameters_Geometry.db
conddb --db myfile.db CTRECO_Geometry_TagXX CTRECO_Geometry_TagXX --destdb CTRECO_Geometry.db
conddb --db myfile.db ZDCRECO_Geometry_TagXX ZDCRECO_Geometry_TagXX --destdb ZDCRECO_Geometry.db
conddb --db myfile.db CASTORRECO_Geometry_TagXX CASTORRECO_Geometry_TagXX --destdb CASTORRECO_Geometry.db
conddb --db myfile.db CSCRECO_Geometry_TagXX CSCRECO_Geometry_TagXX --destdb CSCRECO_Geometry.db
conddb --db myfile.db CSCRECODIGI_Geometry_TagXX CSCRECODIGI_Geometry_TagXX --destdb CSCRECODIGI_Geometry.db
conddb --db myfile.db DTRECO_Geometry_TagXX DTRECO_Geometry_TagXX --destdb DTRECO_Geometry.db
conddb --db myfile.db RPCRECO_Geometry_TagXX RPCRECO_Geometry_TagXX --destdb RPCRECO_Geometry.db
conddb --db myfile.db GEMRECO_Geometry_TagXX GEMRECO_Geometry_TagXX --destdb GEMRECO_Geometry.db
