#ifndef GeometryFCalGeometryShashlikGeometryBuilderFromDDD_h
#define GeometryFCalGeometryShashlikGeometryBuilderFromDDD_h

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

#endif // GeometryFCalGeometryShashlikGeometryBuilderFromDDD_h
