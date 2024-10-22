#!/bin/sh

conddb --yes --db myfile.db copy       PPSRECO_Geometry_2018_TagXX --destdb       PPSRECO_Geometry.db
conddb --yes --db myfile.db copy XMLFILE_CTPPS_Geometry_2018_TagXX --destdb XMLFILE_CTPPS_Geometry.db
