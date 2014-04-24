#ifndef SHASHLIK_GEOMETRY_BUILDER_FROM_DDD_H
# define SHASHLIK_GEOMETRY_BUILDER_FROM_DDD_H

class DDCompactView;
class DDFilteredView;
class ShashlikTopology;
class ShashlikGeometry;

class ShashlikGeometryBuilderFromDDD
{
public:
  ShashlikGeometryBuilderFromDDD( void );
  ~ShashlikGeometryBuilderFromDDD( void );

  ShashlikGeometry* build( const DDCompactView*, const ShashlikTopology& );

private:
  
  ShashlikGeometry* buildGeometry( DDFilteredView&, const ShashlikTopology& );
};

#endif // SHASHLIK_GEOMETRY_BUILDER_FROM_DDD_H
