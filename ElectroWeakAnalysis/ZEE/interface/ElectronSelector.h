std::vector<reco::GsfElectronRef> electronSelector(const std::vector<reco::GsfElectronRef>& electrons,
							EcalClusterLazyTools lazyTools,
							const std::vector<edm::Handle<edm::ValueMap<double> > >& eIsoMap,
							const edm::Handle<trigger::TriggerEvent>& pHLT, const int filterId,
							 const std::vector<double>& Cuts )
{
	std::vector<reco::GsfElectronRef> ChosenOnes;
	const trigger::Keys& ring = pHLT->filterKeys(filterId);
	const trigger::TriggerObjectCollection& HltObjColl = pHLT->getObjects();
	const edm::ValueMap<double>& eIsoMapTrk = *eIsoMap[0];
	const edm::ValueMap<double>& eIsoMapEcal = *eIsoMap[1];
	const edm::ValueMap<double>& eIsoMapHcal = *eIsoMap[2];
	edm::LogDebug_("electronSelector", "", 16)<< "Number of electrons to select from = "<< electrons.size();
	for(std::vector<reco::GsfElectronRef>::const_iterator Relec = electrons.begin(); Relec != electrons.end(); ++Relec)
	{
		reco::GsfElectronRef elec = *Relec;
		edm::LogDebug_("electronSelector", "", 17) <<"Analysing elec, id = "<< elec.id() <<"\tkey = "<< elec.key();
		double scEta = elec->superCluster()->eta();
		if(fabs(scEta) < 1.4442 || fabs(scEta) > 1.56)
		{
			bool HLTMatch = false;
			for(uint k = 0; k < ring.size(); ++k)
			{
				const trigger::TriggerObject& HltObj = HltObjColl[ring[k]];
				if(reco::deltaR(*elec, HltObj) < 0.1) HLTMatch = true; 
			}
			edm::LogDebug_("electronSelector", "", 16) << "HLT Match = "<<HLTMatch;
			std::cout << "HLT Match = " << HLTMatch << std::endl; 
//			if(HLTMatch) ChosenOnes.push_back(elec);
			if(HLTMatch)
			{
				std::vector<float> vCov = lazyTools.localCovariances(*(elec->superCluster()->seed())) ;
				if(fabs(scEta) < 1.479)
				{
					edm::LogDebug_("electronSelector","",32)<<"SigIetaIeta = "<< sqrt(vCov[0])
										<<"\tCut Value = "<< Cuts[1];
					if(sqrt(vCov[0]) < Cuts[1])
					{
					    edm::LogDebug_("elecSel","",39)<<"dEta = "<< elec->deltaEtaSuperClusterTrackAtVtx()
										<<"\tCut Value = "<< Cuts[2];
					    if(fabs(elec->deltaEtaSuperClusterTrackAtVtx()) < Cuts[2])
					    {
					        edm::LogDebug_("elecSel","",39)<<"dPhi = "<< elec->deltaPhiSuperClusterTrackAtVtx()
											<<"\tCut Value = "<< Cuts[3];
					        if(fabs(elec->deltaPhiSuperClusterTrackAtVtx()) < Cuts[3])
					        {
						    edm::LogDebug_("","",29) << "Track isolation = " << eIsoMapTrk[elec] 
										<<"\tCut Value = "<< Cuts[4];
						    if(eIsoMapTrk[elec] < Cuts[4])
						    {				
							edm::LogDebug_("","",29) << "ECAL isolation = " << eIsoMapEcal[elec] 
										<<"\tCut Value = "<< Cuts[5];
							if(eIsoMapEcal[elec] < Cuts[5])
							{
							    edm::LogDebug_("","",29) << "HCAL isolation = " << eIsoMapHcal[elec] 
											<<"\tCut Value = "<< Cuts[6];
								if(eIsoMapHcal[elec] < Cuts[6]) ChosenOnes.push_back(elec);
							}
						    }
						}
					    }
					}
				}else{
					edm::LogDebug_("electronSelector","",32)<<"SigIetaIeta = "<< sqrt(vCov[0])
										<<"\tCut Value = "<< Cuts[7];
					if(sqrt(vCov[0]) < Cuts[7])
					{
					    edm::LogDebug_("elecSel","",39)<<"dEta = "<< elec->deltaEtaSuperClusterTrackAtVtx()
										<<"\tCut Value = "<< Cuts[8];
					    if(fabs(elec->deltaEtaSuperClusterTrackAtVtx()) < Cuts[8])
					    {
					        edm::LogDebug_("elecSel","",39)<<"dPhi = "<< elec->deltaPhiSuperClusterTrackAtVtx()
											<<"\tCut Value = "<< Cuts[9];
					        if(fabs(elec->deltaPhiSuperClusterTrackAtVtx()) < Cuts[9])
					        {
						    edm::LogDebug_("","",29) << "Track isolation = " << eIsoMapTrk[elec] 
										<<"\tCut Value = "<< Cuts[10];
						    if(eIsoMapTrk[elec] < Cuts[10])
						    {				
							edm::LogDebug_("","",29) << "ECAL isolation = " << eIsoMapEcal[elec] 
										<<"\tCut Value = "<< Cuts[11];
							if(eIsoMapEcal[elec] < Cuts[11])
							{
							    edm::LogDebug_("","",29) << "HCAL isolation = " << eIsoMapHcal[elec] 
											<<"\tCut Value = "<< Cuts[12];
								if(eIsoMapHcal[elec] < Cuts[12]) ChosenOnes.push_back(elec);
							}
						    }
						}
					    }
					}
				}
//				ChosenOnes.push_back(elec);
			}
	}
	}
	return ChosenOnes;
}

