#include "CondCore/PopCon/interface/PopConAnalyzer.h"
//
//template <class T>
//popcon::PopConAnalyzer<T>::PopConAnalyzer(const edm::ParameterSet& iConfig)
//{
//
//}
//
//template <class T>
//popcon::PopConAnalyzer<T>::~PopConAnalyzer()
//{	
//	delete m_handler_object;
//}
//
//template <class T>
//void popcon::PopConAnalyzer<T>::takeTheData()
//{
//	m_payload_vect = m_handler_object->returnData();	
//}
//
//template <class T>
//void popcon::PopConAnalyzer<T>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
//{
//	//test - iterate over the payload container
//	//for(std::vector<std::pair<T*,popcon::IOVPair> >::iterator itr = m_payload_vect->begin(); itr != m_payload_vect->end(); itr++)
//	//{	
//	//	//adfsdfa
//	//	std::cout << (*itr)->second.till << std::endl;
//	//}
//	//(std::vector<std::pair<T*,popcon::IOVPair> >)::iterator dupa;
//	int sz = m_payload_vect->size();
//	for(int i=0; i<sz;i++)
//	{
//		std::cout << (*m_payload_vect)[i].second.till;
//	}
//
//}
//
//template <class T>
//void popcon::PopConAnalyzer<T>::beginJob(const edm::EventSetup&)
//{
//	takeTheData();
//}
//
//// ------------ method called once each job just after ending the event loop  ------------
//template <class T>
//void popcon::PopConAnalyzer<T>::endJob() {
//}
