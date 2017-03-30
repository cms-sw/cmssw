#ifndef CondFormats_RPCObjects_RPCDCCLink_h
#define CondFormats_RPCObjects_RPCDCCLink_h

#include <cstdint>
#include <string>
#include <iosfwd>
#include <climits>

#include "CondFormats/Serialization/interface/Serializable.h"

/** Identifier for RPC DCC-LB path */
class RPCDCCLink
{
public:
    static int const wildcard_ = INT_MIN;

    /** @{ */
    /** field ranges */
    static int const min_fed_ = 0;
    static int const max_fed_ = 65534;
    static int const min_dccinput_ = 0;
    static int const max_dccinput_ = 36;
    static int const min_tbinput_ = 0;
    static int const max_tbinput_ = 18;
    /** @} */

protected:
    /** @{ */
    /** field positions and masks */
    static int const pos_fed_ = 16;
    static std::uint32_t const mask_fed_ = 0xffff0000;
    static int const pos_dccinput_ = 8;
    static std::uint32_t const mask_dccinput_ = 0x0000ff00;
    static int const pos_tbinput_ = 0;
    static std::uint32_t const mask_tbinput_ = 0x000000ff;
    /** @} */

public:
    RPCDCCLink();
    RPCDCCLink(std::uint32_t const & id);
    RPCDCCLink(int fed
               , int dccinput
               , int tbinput = wildcard_);

    std::uint32_t getId() const;
    operator std::uint32_t() const;
    std::uint32_t getMask() const;

    bool matches(RPCDCCLink const & rhs) const;

    void setId(std::uint32_t const & id);
    void reset();

    /** @{ */
    /** Field Getters */
    int getFED() const;
    int getDCCInput() const;
    int getTBInput() const;
    /** @} */

    /** @{ */
    /** Field Setters
     * A cms::Exception("OutOfRange") is thrown for out-of-range input values.
     **/
    RPCDCCLink & setFED(int fed = wildcard_);
    RPCDCCLink & setDCCInput(int dccinput = wildcard_);
    RPCDCCLink & setTBInput(int tbinput = wildcard_);
    /** @} */

    std::string getName() const;

    bool operator<(RPCDCCLink const & rhs) const;
    bool operator==(RPCDCCLink const & rhs) const;
    bool operator!=(RPCDCCLink const & rhs) const;
    bool operator<(std::uint32_t const & rhs) const;
    bool operator==(std::uint32_t const & rhs) const;
    bool operator!=(std::uint32_t const & rhs) const;

    RPCDCCLink & operator++();
    RPCDCCLink operator++(int);
    RPCDCCLink & operator--();
    RPCDCCLink operator--(int);

protected:
    int bf_get(int const min, std::uint32_t const mask, int const pos) const;
    RPCDCCLink & bf_set(int const min, int const max, std::uint32_t const mask, int const pos, int const value);
    std::ostream & bf_stream(std::ostream & ostream, int const min, std::uint32_t const mask, int const pos) const;

protected:
    std::uint32_t id_;

    COND_SERIALIZABLE;
};

std::ostream & operator<<(std::ostream & ostream, RPCDCCLink const & link);

#include "CondFormats/RPCObjects/interface/RPCDCCLink.icc"

#endif // CondFormats_RPCObjects_RPCDCCLink_h
