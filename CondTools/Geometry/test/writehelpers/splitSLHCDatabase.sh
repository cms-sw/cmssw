#!/bin/sh
cmscond_export_iov -s sqlite_file:myfile.db -d sqlite_file:GeometryFile_Phase1_R30F12_HCal_Ideal.db -D CondFormatsGeometryObjects  -t XMLFILE_Geometry_TagXX_Phase1_R30F12_HCal_Ideal_mc -l sqlite_file:localpopconlog.db
cmscond_export_iov -s sqlite_file:myfile.db -d sqlite_file:TKRECO_Geometry.db -D CondFormatsGeometryObjects  -t TKRECO_Geometry_TagXX -l sqlite_file:localpopconlog.db
cmscond_export_iov -s sqlite_file:myfile.db -d sqlite_file:TKExtra_Geometry.db -D CondFormatsGeometryObjects  -t TKExtra_Geometry_TagXX -l sqlite_file:localpopconlog.db
cmscond_export_iov -s sqlite_file:myfile.db -d sqlite_file:EBRECO_Geometry.db -D CondFormatsGeometryObjects  -t EBRECO_Geometry_TagXX -l sqlite_file:localpopconlog.db
cmscond_export_iov -s sqlite_file:myfile.db -d sqlite_file:EERECO_Geometry.db -D CondFormatsGeometryObjects  -t EERECO_Geometry_TagXX -l sqlite_file:localpopconlog.db
cmscond_export_iov -s sqlite_file:myfile.db -d sqlite_file:EPRECO_Geometry.db -D CondFormatsGeometryObjects  -t EPRECO_Geometry_TagXX -l sqlite_file:localpopconlog.db
cmscond_export_iov -s sqlite_file:myfile.db -d sqlite_file:HCALRECO_Geometry.db -D CondFormatsGeometryObjects  -t HCALRECO_Geometry_TagXX -l sqlite_file:localpopconlog.db
cmscond_export_iov -s sqlite_file:myfile.db -d sqlite_file:CTRECO_Geometry.db -D CondFormatsGeometryObjects  -t CTRECO_Geometry_TagXX -l sqlite_file:localpopconlog.db
cmscond_export_iov -s sqlite_file:myfile.db -d sqlite_file:ZDCRECO_Geometry.db -D CondFormatsGeometryObjects  -t ZDCRECO_Geometry_TagXX -l sqlite_file:localpopconlog.db
cmscond_export_iov -s sqlite_file:myfile.db -d sqlite_file:CASTORRECO_Geometry.db -D CondFormatsGeometryObjects  -t CASTORRECO_Geometry_TagXX -l sqlite_file:localpopconlog.db
cmscond_export_iov -s sqlite_file:myfile.db -d sqlite_file:CSCRECO_Geometry.db -D CondFormatsGeometryObjects  -t CSCRECO_Geometry_TagXX -l sqlite_file:localpopconlog.db
cmscond_export_iov -s sqlite_file:myfile.db -d sqlite_file:CSCRECODIGI_Geometry.db -D CondFormatsGeometryObjects  -t CSCRECODIGI_Geometry_TagXX -l sqlite_file:localpopconlog.db
cmscond_export_iov -s sqlite_file:myfile.db -d sqlite_file:DTRECO_Geometry.db -D CondFormatsGeometryObjects  -t DTRECO_Geometry_TagXX -l sqlite_file:localpopconlog.db
cmscond_export_iov -s sqlite_file:myfile.db -d sqlite_file:RPCRECO_Geometry.db -D CondFormatsGeometryObjects  -t RPCRECO_Geometry_TagXX -l sqlite_file:localpopconlog.db
