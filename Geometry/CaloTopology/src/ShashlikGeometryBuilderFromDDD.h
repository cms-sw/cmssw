#ifndef SHASHLIK_GEOMETRY_BUILDER_FROM_DDD_H
#define SHASHLIK_GEOMETRY_BUILDER_FROM_DDD_H

class DDCompactView;
class ShashlikTopology;
class ShashlikGeometry;

class ShashlikGeometryBuilderFromDDD
{
public:
  ShashlikGeometryBuilderFromDDD ();
  ~ShashlikGeometryBuilderFromDDD ();

  ShashlikGeometry* build( const DDCompactView*, const ShashlikTopology& );
};

#endif // SHASHLIK_GEOMETRY_BUILDER_FROM_DDD_H
