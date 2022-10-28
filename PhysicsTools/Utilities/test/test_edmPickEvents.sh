#!/bin/bash -ex
#Dataset, Run, Lumi and Events are copied from Workflows 136.8521

export CMS_BOT_USE_DASGOCLIENT=true
edmPickEvents.py --runInteractive "/JetHT/Run2018A-PromptReco-v1/MINIAOD" 315489:31:19015199,315489:31:19098714,315489:31:18897114
