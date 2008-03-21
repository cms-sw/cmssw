#include "CondTools/SiStrip/interface/SiStripConditionHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<iostream>

popcon::SistripConditionsHandler::SistripConditionsHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","SistripConditionsHandler")) {

	std::cout << "SiStripPedestals Source handler constructor\n" << std::endl;
        m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
        m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");

	// qui puoi scrivere altri parametri che devi specificare anche nel .h naturalmente 
	// penso che il minimo sia sid user e pass del DB online 

        std::cout << m_sid<<"/"<<m_user<<"/"<<m_pass   << std::endl;


}

popcon::SistripConditionsHandler::~SistripConditionsHandler()
{
}


void popcon::SistripConditionsHandler::getNewObjects()
{

	std::cout << "------- Ecal - > getNewObjects\n";

	int max_since=0;
	max_since=(int)tagInfo().lastInterval.first;
	std::cout << "max_since : "  << max_since << endl;


	// faccio query al DB online OMDS 
	// recupero tutti i run dopo l'ultimo trafsrito che e` max_since. 
	// chiedere a Domenico o seguo l'esempio che trovo sul twiki per leggere con le librerie CORAL

	cout << "Retrieving run list from ONLINE DB ... " << endl;
	// 


	// qui suppongo di aver recuperato i miei oggetti dal DB online li analizzi
	// li cambio li riformatto... 

	// diciamo che ne ho recuperati n 
	// magari faccio un loop su tutti quelli che ho recuperato da max_since fino a nrun 
	// su indice irun che e` il numero del run di un oggetto specifico
	// dopodiche` li spedisco all'offline
	// definiamo irun=1 per semplicita`

	int irun=1;
	//	 cout << "Generating popcon record for run " << irun << "..." << flush;


	      // suppongo l'oggetto offline e` un  SiStripPedestals e si chiama pedtemp

	      SiStripPedestals* pedtemp = new SiStripPedestals();
	      // nota  dico new qui sopra
	      // popcon cancellera` l'oggetto per me dopo averlo mandato al DB 

	      // giusto per esempio l'oggetto lo riempiamo con valori assurdi 
	      // facciamo un loop su tutti i canali dell'ecal barrel 
	      // e riempiamo un valore per ogni canale 

	      /*for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
		if(iEta==0) continue;
		for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
		  // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
		  if (EBDetId::validDetId(iEta,iPhi))
		    {
		      // questo e` un canale del barrel ad eta e phi specifici iEta e iPhi
		      EBDetId ebdetid(iEta,iPhi);
		      unsigned int hiee = ebdetid.hashedIndex();

		      // questo e` l'oggetto piedistallo per un canale
		      SiStripPedestals::Item item;
		      item.mean_x1  = 0;
		      item.rms_x1   = 1;
		      item.mean_x6  = 2;
		      item.rms_x6   = 3.2;
		      item.mean_x12 = 22;
		      item.rms_x12  = 100;
		      // lo infilo in pedtemp che e` l'oggetto complessivo per tutto il Barrel ECAL
		      // i nostri oggetti sono fatti cosi`... 
		      pedtemp->insert(std::make_pair(ebdetid.rawId(),item));
		     
		    }
		}
	      }*/


	      // qui converto il numero del run in un Time_t  e genero il mio since (snc)

	      Time_t snc= (Time_t) irun ;
	      	      
	      // questa e` l'istruzione che riempie il vettore di oggetti da spedire all'offline 
	      m_to_transfer.push_back(std::make_pair((SiStripPedestals*)pedtemp,snc));
	      
	      // qui avro la fine del mio loop sugli oggetti recuperati dal DB online (OMDS)


	      std::cout << "Ecal - > end of getNewObjects -----------\n";
	      
}	      



