#ifndef TRACKER_GEOMETRY_BUILDER_TRACKER_GEOMETRY_SERVICE_H
# define TRACKER_GEOMETRY_BUILDER_TRACKER_GEOMETRY_SERVICE_H

namespace edm {
  class ActivityRegistry;
  class ParameterSet;
}

class TrackerGeometryService
{
public:
  TrackerGeometryService( edm::ParameterSet const& pset, edm::ActivityRegistry& );
  
  const int rowsPerRoc( void ) const { return m_ROWS_PER_ROC; }
  const int colsPerRoc( void ) const { return m_COLS_PER_ROC; }
  const int bigPixPerRocX( void ) const { return m_BIG_PIX_PER_ROC_X; }
  const int bigPixPerRocY( void ) const { return m_BIG_PIX_PER_ROC_Y; }
  const int rocsX( void ) const { return m_ROCS_X; }
  const int rocsY( void ) const { return m_ROCS_Y; }
  bool upgradeGeometry( void ) const { return m_upgradeGeometry; }
  
private:
  int m_ROWS_PER_ROC;
  int m_COLS_PER_ROC;
  int m_BIG_PIX_PER_ROC_X;
  int m_BIG_PIX_PER_ROC_Y;
  int m_ROCS_X;
  int m_ROCS_Y;
  bool m_upgradeGeometry;
};

#endif // TRACKER_GEOMETRY_BUILDER_TRACKER_GEOMETRY_SERVICE_H
