#!/bin/bash

cmscond_export_iov \
        -s oracle://cms_orcoff_prep/CMS_COND_30X_PIXEL -D CondFormatsSiPixelObjects -d oracle://cms_orcoff_prep/CMS_COND_PIXEL -P /afs/cern.ch/cms/DB/conddb -t SiPixelFedCablingMap_mc
