#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometryService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

TrackerGeometryService::TrackerGeometryService( edm::ParameterSet const& pset, edm::ActivityRegistry& )
  : m_ROWS_PER_ROC( 80 ),     // Num of Rows per ROC 
    m_COLS_PER_ROC( 52 ),     // Num of Cols per ROC
    m_BIG_PIX_PER_ROC_X( 1 ), // in x direction, rows. BIG_PIX_PER_ROC_X = 0 for SLHC
    m_BIG_PIX_PER_ROC_Y( 2 ), // in y direction, cols. BIG_PIX_PER_ROC_Y = 0 for SLHC
    m_ROCS_X( 0 ),	      // 2 for SLHC
    m_ROCS_Y( 0 ),	      // 8 for SLHC
    m_upgradeGeometry( false )
{
  m_ROWS_PER_ROC  = pset.getUntrackedParameter<int>( "ROWS_PER_ROC", m_ROWS_PER_ROC );
  m_COLS_PER_ROC  = pset.getUntrackedParameter<int>( "COLS_PER_ROC", m_COLS_PER_ROC );
  m_BIG_PIX_PER_ROC_X = pset.getUntrackedParameter<int>( "BIG_PIX_PER_ROC_X", m_BIG_PIX_PER_ROC_X );
  m_BIG_PIX_PER_ROC_Y = pset.getUntrackedParameter<int>( "BIG_PIX_PER_ROC_Y", m_BIG_PIX_PER_ROC_Y );
  m_ROCS_X = pset.getUntrackedParameter<int>( "ROCS_X", m_ROCS_X );
  m_ROCS_Y = pset.getUntrackedParameter<int>( "ROCS_Y", m_ROCS_Y );
  m_upgradeGeometry = pset.getUntrackedParameter<bool>( "upgradeGeometry", m_upgradeGeometry );
}

DEFINE_FWK_SERVICE( TrackerGeometryService );
