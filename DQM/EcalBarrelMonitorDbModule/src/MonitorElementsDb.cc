// $Id: MonitorElementsDb.cc,v 1.2 2006/09/22 11:42:00 dellaric Exp $

/*!
  \file MonitorElementsDb.cc
  \brief Generate a Monitor Element from DB data
  \author B. Gobbo 
  \version $Revision: 1.2 $
  \date $Date: 2006/09/22 11:42:00 $
*/

#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>
#include <fstream>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SealKernel/Context.h"
#include "SealKernel/ComponentLoader.h"
#include "SealKernel/Exception.h"
#include "SealKernel/IMessageService.h"
#include "PluginManager/PluginManager.h"
#include "RelationalAccess/IConnectionService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"

#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"

#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"

#include "TROOT.h"
#include "TStyle.h"
#include "TPaveStats.h"

#include <DQM/EcalBarrelMonitorDbModule/interface/MonitorElementsDb.h>


MonitorElementsDb::MonitorElementsDb( const edm::ParameterSet& ps, std::string xmlFile ){

  xmlFile_ = xmlFile;

  // get hold of back-end interface
  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();

  if ( dbe_ ) {

    dbe_->setCurrentFolder("EcalBarrel/MonitorElementsDb");

    //meTemp_ = dbe_->book2D("TEMP", "TEMP", 17, -0.5, 16.5, 10, -0.5, 9.5);

    parser_ = new MonitorXMLParser( xmlFile_ );
    try {
      parser_->load();
    } catch( const std::runtime_error e ) {
      std::cerr << "Error loading parser: " << e.what() << std::endl;
    }

    MEinfo_ = parser_->getDB_ME();

    for( unsigned int i=0; i< MEinfo_.size(); i++ ) {
      
      MonitorElement* tmp;
      tmp = 0;
      if( MEinfo_[i].type == "th1d" ) {
	tmp = dbe_->book1D( MEinfo_[i].title, MEinfo_[i].title, MEinfo_[i].xbins, MEinfo_[i].xfrom, MEinfo_[i].xto );
      }
      else if( MEinfo_[i].type == "th2d" ) {
	tmp = dbe_->book2D( MEinfo_[i].title, MEinfo_[i].title, MEinfo_[i].xbins, MEinfo_[i].xfrom, MEinfo_[i].xto,
			    MEinfo_[i].ybins, MEinfo_[i].yfrom, MEinfo_[i].yto );
      }
      else if( MEinfo_[i].type == "tprofile" ) {
      tmp = dbe_->bookProfile( MEinfo_[i].title, MEinfo_[i].title, MEinfo_[i].xbins, MEinfo_[i].xfrom, MEinfo_[i].xto,
			       MEinfo_[i].ybins, MEinfo_[i].yfrom, MEinfo_[i].yto );
      }
      else if( MEinfo_[i].type == "tprofile2d" ) {
	tmp = dbe_->bookProfile2D( MEinfo_[i].title, MEinfo_[i].title, MEinfo_[i].xbins, MEinfo_[i].xfrom, MEinfo_[i].xto,
				   MEinfo_[i].ybins, MEinfo_[i].yfrom, MEinfo_[i].yto, 
				   MEinfo_[i].zbins, MEinfo_[i].zfrom, MEinfo_[i].zto );
      }
      
      MEs_.push_back( tmp );
    }

  }

}

MonitorElementsDb::~MonitorElementsDb(){

}

void MonitorElementsDb::beginJob(const edm::EventSetup& c){

  ievt_ = 0;
    
}

void MonitorElementsDb::endJob( void ){

  std::cout << "MonitorElementsDb: analyzed " << ievt_ << " events" << std::endl;
  for( unsigned int i = 0; i<MEs_.size(); i++ ) {
    if( MEs_[i] != 0 ) dbe_->removeElement( MEs_[i]->getName() );
  }

}

void MonitorElementsDb::analyze( const edm::Event& e, const edm::EventSetup& c, coral::ISessionProxy* session ){

  ievt_++;

  bool atLeastAQuery;
  atLeastAQuery = false;

  std::vector<std::string> vars;

  if ( session )  {

    for( unsigned int i=0; i<MEinfo_.size(); i++ ) {

      // i-th ME...

      if( MEs_[i] != 0 && ( ievt_ % MEinfo_[i].ncycle ) == 0 ) {

	vars.clear();

	try {

	  atLeastAQuery = true;

	  session->transaction().start(true);

	  coral::ISchema& schema = session->nominalSchema();

	  coral::IQuery* query = schema.newQuery();

	  for( unsigned int j=0; j<MEinfo_[i].queries.size(); j++ ) {
	    if( MEinfo_[i].queries[j].query == "addToTableList" ) {
	      query->addToTableList( MEinfo_[i].queries[j].arg );
	    }
	    else if( MEinfo_[i].queries[j].query == "addToOutputList" ) {
	      query->addToOutputList( MEinfo_[i].queries[j].arg, MEinfo_[i].queries[j].alias );
	      vars.push_back( MEinfo_[i].queries[j].alias );
	    }
	    else if( MEinfo_[i].queries[j].query == "setCondition" ) {
	      query->setCondition( MEinfo_[i].queries[j].arg, coral::AttributeList() );
	    }
	    else if( MEinfo_[i].queries[j].query == "addToOrderList" ) {
	      query->addToOrderList( MEinfo_[i].queries[j].arg );
	    }
	  }
	  
	  coral::ICursor& cursor = query->execute();

	  // pause the shipping of monitoring elements
	  if ( dbe_ ) dbe_->lock();
	  
	  unsigned int k = 0;
	
	  while ( cursor.next() && k < MEinfo_[i].loop ) {
	    //while ( cursor.next() ) {
	  
	    const coral::AttributeList& row = cursor.currentRow();
	    
	    std::vector<float> vvars;
            vvars.clear();
	    for( unsigned int l=0; l<vars.size(); l++ ) {
	      if( !vars[l].empty() ) {
		vvars.push_back( row[vars[l].c_str()].data<float>() );  
	      }
	    }
	    if( vvars.size() == 2 ) {
	      //std::cout << k << " -- " << vvars[0] << " -- " << vvars[1] << std::endl;
	      MEs_[i]->Fill( vvars[0], vvars[1] );
	    }
	    else if( vvars.size() == 3 ) {
	      //std::cout << k << " -- " << vvars[0] << " -- " << vvars[1] << " -- " << vvars[2] << std::endl;
	      MEs_[i]->Fill( vvars[0], vvars[1], vvars[2] );
	    }
	    else if( vvars.size() == 4 ) {
	      //std::cout << k << " -- " << vvars[0] << " -- " << vvars[1] << " -- " << vvars[2] << " -- " << vvars[3] << std::endl;
	      MEs_[i]->Fill( vvars[0], vvars[1], vvars[2], vvars[3] );
	    }
	    else{
	      std::cerr << "Too many variables to plot..." << std::endl;
	      exit(1);
	    }
	    	  
	    k++;
	    
	  }
	  
	  // resume the shipping of monitoring elements
	  if ( dbe_ ) dbe_->unlock();
	  
	  delete query;
	  
	} catch (coral::Exception& se) {
	  std::cerr << "CORAL Exception : " << se.what() << std::endl;
	} catch (std::exception& e) {
	  std::cerr << "Standard C++ exception : " << e.what() << std::endl;
	} catch (...) {
	  std::cerr << "Exception caught (...)" << std::endl;
	}


      }

    }

    if( atLeastAQuery ) session->transaction().commit();

  }

}

void MonitorElementsDb::htmlOutput(std::string htmlDir){

  gStyle->SetOptStat(0);
  gStyle->SetOptFit();
  gStyle->SetPalette(1,0);
  
  for( unsigned int i=0; i<MEinfo_.size(); i++ ) {

    if( MEs_[i] != 0 && ( ievt_ % MEinfo_[i].ncycle ) == 0 ) {

      TCanvas* c1;
      int n = MEinfo_[i].xbins > MEinfo_[i].ybins ? int( round( float( MEinfo_[i].xbins ) / float( MEinfo_[i].ybins ) ) ) :
	int( round( float( MEinfo_[i].ybins ) / float( MEinfo_[i].xbins ) ) );
      if( MEinfo_[i].xbins > MEinfo_[i].ybins ) {
	c1 = new TCanvas( "c1", "dummy", 400*n, 400 );
      }
      else {
	c1 = new TCanvas( "c1", "dummy", 400, 400*n );
      }
      c1->SetGrid();
      c1->cd();
    
      const double histMax = 1.e15;

      MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*> (MEs_[i]);
      if ( ob ) {
	if( dynamic_cast<TH1F*>( ob->operator->()) ) {
	  TH1F* h = dynamic_cast<TH1F*> (ob->operator->());
	  h->Draw( );
	}
	else if( dynamic_cast<TH2F*>( ob->operator->()) ) {
	  TH2F* h = dynamic_cast<TH2F*>( ob->operator->() );
	  if( h->GetMaximum(histMax) > 1.e4 ) {
	    gPad->SetLogz(1);
	  } else {
	    gPad->SetLogz(0);
	  }
	  h->Draw( "colz" );
	}
	else if( dynamic_cast<TProfile*>( ob->operator->()) ) {
	  TProfile* h = dynamic_cast<TProfile*>( ob->operator->() );
	  if( h->GetMaximum(histMax) > 1.e4 ) {
	    gPad->SetLogz(1);
	  } else {
	    gPad->SetLogz(0);
	  }
	  h->Draw( "colz" );
	}
      }

      c1->Update();
      std::string name = htmlDir + "/" + MEinfo_[i].title + ".png"; 
      c1->SaveAs( name.c_str() );
  
      delete c1;

    }
  }
}

