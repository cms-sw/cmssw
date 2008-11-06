#ifndef Fireworks_Vertices_VerticesProxy3DBuilder_h
#define Fireworks_Vertices_VerticesProxy3DBuilder_h

#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class VerticesProxy3DBuilder : public FWRPZDataProxyBuilder {
public:
    VerticesProxy3DBuilder();
    virtual ~VerticesProxy3DBuilder();

    REGISTER_PROXYBUILDER_METHODS();

private:
    virtual void build (const FWEventItem* item, TEveElementList** product);

    // prevent default copy constructor and assignment operator
    VerticesProxy3DBuilder (const VerticesProxy3DBuilder &);
    const VerticesProxy3DBuilder & operator=(const VerticesProxy3DBuilder &);
};

#endif // Fireworks_Vertices_VerticesProxy3DBuilder_h
