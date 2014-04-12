#ifndef GENERS_ITEMLOCATION_HH_
#define GENERS_ITEMLOCATION_HH_

#include <string>
#include <iostream>

#include "Alignment/Geners/interface/ClassId.hh"

namespace gs {
    class ItemLocation
    {
    public:
        inline ItemLocation(std::streampos pos, const char* URI,
                            const char* cachedItemURI=0)
            : pos_(pos),
              URI_(URI ? URI : ""),
              cachedItemURI_(cachedItemURI ? cachedItemURI : "") {}

        inline std::streampos streamPosition() const {return pos_;}
        inline const std::string& URI() const {return URI_;}
        inline const std::string& cachedItemURI() const
           {return cachedItemURI_;}

        inline void setStreamPosition(std::streampos pos) {pos_ = pos;}
        inline void setURI(const char* newURI)
           {URI_ = newURI ? newURI : "";}
        inline void setCachedItemURI(const char* newURI)
           {cachedItemURI_ = newURI ? newURI : "";}

        bool operator==(const ItemLocation& r) const;
        inline bool operator!=(const ItemLocation& r) const
            {return !(*this == r);}

        // Methods related to I/O
        inline ClassId classId() const {return ClassId(*this);}
        bool write(std::ostream& of) const;

        static inline const char* classname() {return "gs::ItemLocation";}
        static inline unsigned version() {return 1;}
        static ItemLocation* read(const ClassId& id, std::istream& in);

    private:
        ItemLocation();

        std::streampos pos_;        
        std::string URI_;
        std::string cachedItemURI_;
    };
}

#endif // GENERS_ITEMLOCATION_HH_

