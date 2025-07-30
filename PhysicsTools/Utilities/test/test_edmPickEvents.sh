#!/bin/bash -ex
#Dataset, Run, Lumi and Events are copied from Workflows 136.8521

export CMS_BOT_USE_DASGOCLIENT=true
edmPickEvents.py --runInteractive "/Muon/Run2022G-22Sep2023-v1/MINIAOD" 362439:809:1727980388 362439:896:1918814501 362439:1019:2181752649
