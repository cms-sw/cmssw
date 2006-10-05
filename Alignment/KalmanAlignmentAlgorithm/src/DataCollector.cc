
#include "Alignment/KalmanAlignmentAlgorithm/interface/DataCollector.h"

#include <iostream>
#include <algorithm>

#include "TH1F.h"
#include "TGraph.h"
#include "TFile.h"

using namespace alignmentservices;

#ifdef SERVICE_WORKAROUND
DataCollector* DataCollector::theDataCollector = 0;
#endif


DataCollector::DataCollector( void ) {}


DataCollector::DataCollector( const edm::ParameterSet & config ) : theConfiguration( config ) {}


DataCollector::~DataCollector() {}


bool DataCollector::isAvailable( void )
{
#ifdef SERVICE_WORKAROUND
  return true;
#else
  return edm::Service< DataCollector >().isAvailable();
#endif
}


DataCollector* DataCollector::get( void )
{
#ifdef SERVICE_WORKAROUND
  if ( !theDataCollector ) theDataCollector = new DataCollector();
  return theDataCollector;
#else
  return edm::Service< DataCollector >().operator->();
#endif
}


void DataCollector::fillHistogram( string histo_name, float data )
{
#ifdef SERVICE_WORKAROUND
  if ( !theDataCollector ) theDataCollector = new DataCollector();
  theDataCollector->fillTH1F( histo_name, data );
#else
  if ( isAvilable() ) edm::Service< DataCollector >()->fillTH1F( histo_name, data );
  else notify();
#endif
}


void DataCollector::fillHistogram( string histo_name, int histo_number, float data )
{
#ifdef SERVICE_WORKAROUND
  if ( !theDataCollector ) theDataCollector = new DataCollector();
  theDataCollector->fillTH1F( histo_name, histo_number, data );
#else
  if ( isAvilable() ) edm::Service< DataCollector >()->fillTH1F( histo_name, histo_number, data );
  else notify();
#endif
}


void DataCollector::fillGraph( string graph_name, float x_data, float y_data )
{
#ifdef SERVICE_WORKAROUND
  if ( !theDataCollector ) theDataCollector = new DataCollector();
  theDataCollector->fillTGraph( graph_name, x_data, y_data );
#else
  if ( isAvilable() ) edm::Service< DataCollector >()->fillTGraph( graph_name, x_data, y_data );
  else notify();
#endif
}


void DataCollector::fillGraph( string graph_name, int graph_number, float x_data, float y_data )
{
#ifdef SERVICE_WORKAROUND
  if ( !theDataCollector ) theDataCollector = new DataCollector();
  theDataCollector->fillTGraph( graph_name, graph_number, x_data, y_data );
#else
  if ( isAvilable() ) edm::Service< DataCollector >()->fillTGraph( graph_name, graph_number, x_data, y_data );
  else notify();
#endif
}


void DataCollector::write( string file_name, string mode )
{
#ifdef SERVICE_WORKAROUND
  if ( !theDataCollector ) theDataCollector = new DataCollector();
  theDataCollector->writeToTFile( file_name, mode );
#else
  if ( isAvilable() ) edm::Service< DataCollector >()->writeToTFile( file_name, mode );
  else notify();
#endif
}


void DataCollector::clear( void )
{
#ifdef SERVICE_WORKAROUND
  if ( theDataCollector ) theDataCollector->clear();
#else
  if ( isAvilable() ) edm::Service< DataCollector >()->clear();
  else notify();
#endif
}


void DataCollector::fillTH1F( string histo_name, float data )
{
  if ( theHistoData.find( histo_name ) == theHistoData.end() )
  {
    theHistoData[histo_name] = vector< float > ( 1, data );
  }
  else
  {
    theHistoData[histo_name].push_back( data );
  }

  return;
}


void DataCollector::fillTH1F( string histo_name, int histo_number, float data )
{
  string full_histo_name = histo_name + toString( histo_number );
  fillTH1F( full_histo_name, data );

  return;
}


void DataCollector::fillTGraph( string graph_name, float x_data, float y_data )
{
  if ( theXGraphData.find( graph_name ) == theXGraphData.end() )
  {
    theXGraphData[graph_name] = vector< float > ( 1, x_data );
    theYGraphData[graph_name] = vector< float > ( 1, y_data );
  }
  else
  {
    theXGraphData[graph_name].push_back( x_data );
    theYGraphData[graph_name].push_back( y_data );
  }

  return;
}


void DataCollector::fillTGraph( string graph_name, int graph_number, float x_data, float y_data )
{
  string full_graph_name = graph_name + toString( graph_number );
  fillTGraph( full_graph_name, x_data, y_data );

  return;
}


void DataCollector::writeToTFile( string file_name, string mode )
{
  int nBins = theConfiguration.getUntrackedParameter< int >( "NBins", 200 );
  double xMin = theConfiguration.getUntrackedParameter< double >( "XMin", -10. );
  double xMax = theConfiguration.getUntrackedParameter< double >( "XMax", 10. );

  TFile* file = new TFile( file_name.c_str(), mode.c_str() );

  if ( !theHistoData.empty() )
  {
    map< string, vector< float > >::iterator itH = theHistoData.begin();
    vector< float >::iterator itV;
    TH1F* tempHisto;

    while ( itH != theHistoData.end() )
    {
      tempHisto = new TH1F( itH->first.c_str(), itH->first.c_str(), nBins, xMin, xMax );

      itV = itH->second.begin();
      while ( itV != itH->second.end() ) {
	tempHisto->Fill( *itV );
	itV++;
      }

      tempHisto->Write();
      delete tempHisto;

      itH++;
    }
  }

  if ( !theXGraphData.empty() )
  {
    map< string, vector< float > >::iterator itXG = theXGraphData.begin();
    map< string, vector< float > >::iterator itYG = theYGraphData.begin();

    float* xData;
    float* yData;

    TGraph* tempGraph;

    while ( itXG != theXGraphData.end() )
    {
      int nData = itXG->second.size();

      xData = new float[nData];
      yData = new float[nData];

      for ( int iData = 0; iData < nData; iData++ )
      {
	xData[iData] = itXG->second[iData];
	yData[iData] = itYG->second[iData];
      }

      tempGraph = new TGraph( nData, xData, yData );
      tempGraph->SetName( itXG->first.c_str() );
      tempGraph->SetTitle( itXG->first.c_str() );

      tempGraph->Write();
      delete tempGraph;

      delete[] xData;
      delete[] yData;

      itXG++;
      itYG++;
    }
  }

  file->Write();
  file->Close();
  delete file;

  return;
}


void DataCollector::clearData( void )
{
  theHistoData.clear();
  theXGraphData.clear();
  theYGraphData.clear();
}


string DataCollector::toString( int i )
{
  char temp[10];
  sprintf( temp, "%u", i );

  return string( temp );
}


void DataCollector::notify( void ) { cout << "[DataCollector] Service not available." << endl; }
