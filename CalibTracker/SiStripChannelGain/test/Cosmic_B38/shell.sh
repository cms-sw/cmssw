export STAGE_SVCCLASS=cmscaf
export STAGER_TRACE=3
#nsls /castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/
echo "replace PoolSource.fileNames = {"
nsls /castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0000/ | sed -e "s+^+'rfio:/castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0000/+" | sed -e "s+\$+',+"
nsls /castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0001/ | sed -e "s+^+'rfio:/castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0001/+" | sed -e "s+\$+',+"
nsls /castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0002/ | sed -e "s+^+'rfio:/castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0002/+" | sed -e "s+\$+',+"
nsls /castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0003/ | sed -e "s+^+'rfio:/castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0003/+" | sed -e "s+\$+',+"
nsls /castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0004/ | sed -e "s+^+'rfio:/castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0004/+" | sed -e "s+\$+',+"
nsls /castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0005/ | sed -e "s+^+'rfio:/castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0005/+" | sed -e "s+\$+',+"
nsls /castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0009/ | sed -e "s+^+'rfio:/castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0009/+" | sed -e "s+\$+',+"
nsls /castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0026/ | sed -e "s+^+'rfio:/castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0026/+" | sed -e "s+\$+',+"
nsls /castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0042/ | sed -e "s+^+'rfio:/castor/cern.ch/cms/store/mc/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/0042/+" | sed -e "s+\$+',+"
echo "}"



