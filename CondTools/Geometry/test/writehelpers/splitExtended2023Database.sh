#!/bin/sh

conddb --yes --db myfile.db copy XMLFILE_Geometry_TagXX_Extended2023_mc --destdb GeometryFileExtended2023.db
conddb --yes --db myfile.db copy XMLFILE_Geometry_TagXX_Extended2023ZeroMaterial_mc --destdb GeometryFileExtended2023ZeroMaterial.db
conddb --yes --db myfile.db copy XMLFILE_Geometry_TagXX_Extended2023FlatMinus05Percent_mc --destdb GeometryFileExtended2023FlatMinus05Percent.db
conddb --yes --db myfile.db copy XMLFILE_Geometry_TagXX_Extended2023FlatMinus10Percent_mc --destdb GeometryFileExtended2023FlatMinus10Percent.db
conddb --yes --db myfile.db copy XMLFILE_Geometry_TagXX_Extended2023FlatPlus05Percent_mc --destdb GeometryFileExtended2023FlatPlus05Percent.db
conddb --yes --db myfile.db copy XMLFILE_Geometry_TagXX_Extended2023FlatPlus10Percent_mc --destdb GeometryFileExtended2023FlatPlus10Percent.db
conddb --yes --db myfile.db copy TKRECO_Geometry_TagXX                  --destdb TKRECO_Geometry.db
conddb --yes --db myfile.db copy TKParameters_Geometry_TagXX            --destdb TKParameters_Geometry.db
conddb --yes --db myfile.db copy EBRECO_Geometry_TagXX                  --destdb EBRECO_Geometry.db
conddb --yes --db myfile.db copy EERECO_Geometry_TagXX                  --destdb EERECO_Geometry.db
conddb --yes --db myfile.db copy EPRECO_Geometry_TagXX                  --destdb EPRECO_Geometry.db
conddb --yes --db myfile.db copy HCALRECO_Geometry_TagXX                --destdb HCALRECO_Geometry.db
conddb --yes --db myfile.db copy HCALParameters_Geometry_TagXX          --destdb HCALParameters_Geometry.db
conddb --yes --db myfile.db copy CTRECO_Geometry_TagXX                  --destdb CTRECO_Geometry.db
conddb --yes --db myfile.db copy ZDCRECO_Geometry_TagXX                 --destdb ZDCRECO_Geometry.db
conddb --yes --db myfile.db copy CSCRECO_Geometry_TagXX                 --destdb CSCRECO_Geometry.db
conddb --yes --db myfile.db copy CSCRECODIGI_Geometry_TagXX             --destdb CSCRECODIGI_Geometry.db
conddb --yes --db myfile.db copy DTRECO_Geometry_TagXX                  --destdb DTRECO_Geometry.db
conddb --yes --db myfile.db copy RPCRECO_Geometry_TagXX                 --destdb RPCRECO_Geometry.db
conddb --yes --db myfile.db copy GEMRECO_Geometry_TagXX                 --destdb GEMRECO_Geometry.db
