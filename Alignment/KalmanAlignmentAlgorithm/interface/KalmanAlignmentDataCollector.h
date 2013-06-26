#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentDataCollector_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentDataCollector_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <map>
#include <string>

/// A simple class that allows fast and easy histograming and the production of graphs.

class KalmanAlignmentDataCollector
{

public:

  KalmanAlignmentDataCollector( void );
  KalmanAlignmentDataCollector( const edm::ParameterSet& config );
  ~KalmanAlignmentDataCollector( void );

  static KalmanAlignmentDataCollector* get( void );

  static void configure( const edm::ParameterSet& config );

  static void fillHistogram( std::string histo_name, float data );
  static void fillHistogram( std::string histo_name, int histo_number, float data );

  static void fillGraph( std::string graph_name, float x_data, float y_data );
  static void fillGraph( std::string graph_name, int graph_number, float x_data, float y_data );

  static void fillNtuple( std::string ntuple_name, float data );

  static void write( void );
  static void write( std::string file_name, std::string mode = "RECREATE" );

  static void clear( void );
    
private:

  void config( const edm::ParameterSet & config );

  void fillTH1F( std::string histo_name, float data );
  void fillTH1F( std::string histo_name, int histo_number, float data );
  
  void fillTGraph( std::string graph_name, float x_data, float y_data );
  void fillTGraph( std::string graph_name, int graph_number, float x_data, float y_data );

  void fillTNtuple( std::string ntuple_name, float data );
  
  void writeToTFile( void );
  void writeToTFile( std::string file_name, std::string mode = "RECREATE" );
  
  void clearData( void );

  std::string toString( int );
  
  static KalmanAlignmentDataCollector* theDataCollector;

  edm::ParameterSet theConfiguration;
  
  std::map< std::string, std::vector< float > > theHistoData;
  std::map< std::string, std::vector< float > > theXGraphData;
  std::map< std::string, std::vector< float > > theYGraphData;
  std::map< std::string, std::vector< float > > theNtupleData;
};


#endif
