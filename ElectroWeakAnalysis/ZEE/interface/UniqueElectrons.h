std::vector<reco::GsfElectronRef> uniqueElectronFinder(edm::Handle<reco::GsfElectronCollection>& pElectrons)
{
	const reco::GsfElectronCollection *electrons = pElectrons.product();
	//Remove duplicate electrons which share a supercluster
        std::vector<reco::GsfElectronRef> UniqueElectrons;
	int index =0;
	for(reco::GsfElectronCollection::const_iterator elec = electrons->begin(); elec != electrons->end();++elec)
    	{
        	const reco::GsfElectronRef electronRef(pElectrons, index);
        	reco::GsfElectronCollection::const_iterator BestDuplicate = elec;
	        for(reco::GsfElectronCollection::const_iterator elec2 = electrons->begin(); elec2 != electrons->end(); ++elec2)
        	{
	            if(elec != elec2)
	            {
	                if(elec->superCluster() == elec2->superCluster())
        	        {
	                    edm::LogDebug_("", "MySelection.cc", 122)<<"e/p Best duplicate = "<< BestDuplicate->eSuperClusterOverP()
											 <<"\telec2 = "<<elec2->eSuperClusterOverP();
        	            if(fabs(BestDuplicate->eSuperClusterOverP()-1.) >= fabs(elec2->eSuperClusterOverP()-1.))
                	    {
                        	    BestDuplicate = elec2;
	                            edm::LogDebug_("", "MySelection.cc", 122)<<"elec2 is now best duplicate";
        	            }else edm::LogDebug_("", "MySelection.cc", 122)<<"BestDuplicate remains best duplicate";
                	 }
	            }
        	 }
                 if(BestDuplicate == elec) UniqueElectrons.push_back(electronRef);
	         ++index;
     	}	
	return UniqueElectrons;
}

