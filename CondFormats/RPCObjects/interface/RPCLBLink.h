#ifndef CondFormats_RPCObjects_RPCLBLink_h
#define CondFormats_RPCObjects_RPCLBLink_h

#include <cstdint>
#include <string>
#include <iosfwd>
#include <climits>

#include "CondFormats/Serialization/interface/Serializable.h"

/** Identifier for RPC Readout Links, following CMS IN 2002/065 */
class RPCLBLink
{
public:
    static int const wildcard_ = INT_MIN;

    /** @{ */
    /** field ranges */
    static int const min_region_ = 0;
    static int const max_region_ = 1;
    static int const min_yoke_ = -5;
    static int const max_yoke_ = 5;
    static int const min_sector_ = 1;
    static int const max_sector_ = 12;
    static int const min_side_ = 0;
    static int const max_side_ = 2;
    static int const min_wheelordisk_ = 0;
    static int const max_wheelordisk_ = 4;
    static int const min_fibre_ = 0;
    static int const max_fibre_ = 7;
    static int const min_radial_ = 0;
    static int const max_radial_ = 1;
    static int const min_linkboard_ = 0;
    static int const max_linkboard_ = 2;
    static int const min_connector_ = 0;
    static int const max_connector_ = 5;
    /** @} */

protected:
    /** @{ */
    /** field positions and masks */
    static int const pos_region_ = 24;
    static std::uint32_t const mask_region_ = 0x03000000;
    static int const pos_yoke_ = 20;
    static std::uint32_t const mask_yoke_ = 0x00f00000;
    static int const pos_sector_ = 16;
    static std::uint32_t const mask_sector_ = 0x000f0000;
    static int const pos_side_ = 14;
    static std::uint32_t const mask_side_ = 0x0000c000;
    static int const pos_wheelordisk_ = 11;
    static std::uint32_t const mask_wheelordisk_ = 0x00003800;
    static int const pos_fibre_ = 7;
    static std::uint32_t const mask_fibre_ = 0x00000780;
    static int const pos_radial_ = 5;
    static std::uint32_t const mask_radial_ = 0x00000060;
    static int const pos_linkboard_ = 3;
    static std::uint32_t const mask_linkboard_ = 0x00000018;
    static int const pos_connector_ = 0;
    static std::uint32_t const mask_connector_ = 0x00000007;
    /** @} */

public:
    RPCLBLink();
    RPCLBLink(std::uint32_t const & id);
    RPCLBLink(int region
              , int yoke
              , int sector = wildcard_
              , int side = wildcard_
              , int wheelordisk = wildcard_
              , int fibre = wildcard_
              , int radial = wildcard_
              , int linkboard = wildcard_
              , int connector = wildcard_);

    std::uint32_t getId() const;
    operator std::uint32_t() const;
    std::uint32_t getMask() const;

    bool matches(RPCLBLink const & rhs) const;

    void setId(std::uint32_t const & id);
    void reset();

    /** @{ */
    /** Field Getters */
    int getRegion() const;
    int getYoke() const;
    int getSector() const;
    int getSide() const;
    int getWheelOrDisk() const;
    int getFibre() const;
    int getRadial() const;
    int getLinkBoard() const;
    int getConnector() const;
    /** @} */

    /** @{ */
    /** Field Setters
     * A cms::Exception("OutOfRange") is thrown for out-of-range input values.
     **/
    RPCLBLink & setRegion(int region = wildcard_);
    RPCLBLink & setYoke(int yoke = wildcard_);
    RPCLBLink & setSector(int sector = wildcard_);
    RPCLBLink & setSide(int side = wildcard_);
    RPCLBLink & setWheelOrDisk(int wheelordisk = wildcard_);
    RPCLBLink & setFibre(int fibre = wildcard_);
    RPCLBLink & setRadial(int radial = wildcard_);
    RPCLBLink & setLinkBoard(int linkboard = wildcard_);
    RPCLBLink & setConnector(int connector = wildcard_);
    /** @} */

    std::string getName() const;

    bool operator<(RPCLBLink const & rhs) const;
    bool operator==(RPCLBLink const & rhs) const;
    bool operator!=(RPCLBLink const & rhs) const;
    bool operator<(std::uint32_t const & rhs) const;
    bool operator==(std::uint32_t const & rhs) const;
    bool operator!=(std::uint32_t const & rhs) const;

    RPCLBLink & operator++();
    RPCLBLink operator++(int);
    RPCLBLink & operator--();
    RPCLBLink operator--(int);

protected:
    int bf_get(int const min, std::uint32_t const mask, int const pos) const;
    RPCLBLink & bf_set(int const min, int const max, std::uint32_t const mask, int const pos, int const value);
    std::ostream & bf_stream(std::ostream & ostream, int const min, std::uint32_t const mask, int const pos) const;

protected:
    std::uint32_t id_;

    COND_SERIALIZABLE;
};

std::ostream & operator<<(std::ostream & ostream, RPCLBLink const & link);

#include "CondFormats/RPCObjects/interface/RPCLBLink.icc"

#endif // CondFormats_RPCObjects_RPCLBLink_h
