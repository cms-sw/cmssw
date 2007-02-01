#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentDataCollector_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentDataCollector_h

// !!! workaround until services work for loopers !!!
#ifndef SERVICE_WORKAROUND
#define SERVICE_WORKAROUND
#endif

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

#include <vector>
#include <map>
#include <string>

/// A simple class that allows fast and easy histograming and the production of graphs. It is in principle working
/// like a service - unfortunately the usual services are not available from within an ESProducerLooper, such that 
/// a workaround must do for the time being. 

using namespace std;

class KalmanAlignmentDataCollector
{

public:

  KalmanAlignmentDataCollector( void );
  KalmanAlignmentDataCollector( const edm::ParameterSet& config );
  ~KalmanAlignmentDataCollector( void );

  static KalmanAlignmentDataCollector* get( void );

  static void configure( const edm::ParameterSet& config );

  static void fillHistogram( string histo_name, float data );
  static void fillHistogram( string histo_name, int histo_number, float data );

  static void fillGraph( string graph_name, float x_data, float y_data );
  static void fillGraph( string graph_name, int graph_number, float x_data, float y_data );

  static void write( void );
  static void write( string file_name, string mode = "RECREATE" );

  static void clear( void );
    
private:

  void config( const edm::ParameterSet & config );

  void fillTH1F( string histo_name, float data );
  void fillTH1F( string histo_name, int histo_number, float data );
  
  void fillTGraph( string graph_name, float x_data, float y_data );
  void fillTGraph( string graph_name, int graph_number, float x_data, float y_data );
  
  void writeToTFile( void );
  void writeToTFile( string file_name, string mode = "RECREATE" );
  
  void clearData( void );

  string toString( int );
  
  static KalmanAlignmentDataCollector* theDataCollector;

  edm::ParameterSet theConfiguration;
  
  map< string, vector< float > > theHistoData;
  map< string, vector< float > > theXGraphData;
  map< string, vector< float > > theYGraphData;

};


#endif
