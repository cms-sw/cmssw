#ifndef CondFormats_RPCObjects_RPCAMCLink_h
#define CondFormats_RPCObjects_RPCAMCLink_h

#include <cstdint>
#include <string>
#include <iosfwd>
#include <climits>

#include "CondFormats/Serialization/interface/Serializable.h"

/** Identifier for RPC AMC-LB path */
class RPCAMCLink
{
public:
    static int const wildcard_ = INT_MIN;

    /** @{ */
    /** field ranges */
    static int const min_fed_ = 0;
    static int const max_fed_ = 65534;
    static int const min_amcnumber_ = 0;
    static int const max_amcnumber_ = 12;
    static int const min_amcinput_ = 0;
    static int const max_amcinput_ = 80;
    /** @} */

protected:
    /** @{ */
    /** field positions and masks */
    static int const pos_fed_ = 16;
    static std::uint32_t const mask_fed_ = 0xffff0000;
    static int const pos_amcnumber_ = 12;
    static std::uint32_t const mask_amcnumber_ = 0x0000f000;
    static int const pos_amcinput_ = 0;
    static std::uint32_t const mask_amcinput_ = 0x00000fff;
    /** @} */

public:
    RPCAMCLink();
    RPCAMCLink(std::uint32_t const & id);
    RPCAMCLink(int fed
               , int amcnumber
               , int amcinput = wildcard_);

    std::uint32_t getId() const;
    operator std::uint32_t() const;
    std::uint32_t getMask() const;

    bool matches(RPCAMCLink const & rhs) const;

    void setId(std::uint32_t const & id);
    void reset();

    /** @{ */
    /** Field Getters */
    int getFED() const;
    int getAMCNumber() const;
    int getAMCInput() const;
    /** @} */

    /** @{ */
    /** Field Setters
     * A cms::Exception("OutOfRange") is thrown for out-of-range input values.
     **/
    RPCAMCLink & setFED(int fed = wildcard_);
    RPCAMCLink & setAMCNumber(int amcnumber = wildcard_);
    RPCAMCLink & setAMCInput(int amcinput = wildcard_);
    /** @} */

    std::string getName() const;

    bool operator<(RPCAMCLink const & rhs) const;
    bool operator==(RPCAMCLink const & rhs) const;
    bool operator!=(RPCAMCLink const & rhs) const;
    bool operator<(std::uint32_t const & rhs) const;
    bool operator==(std::uint32_t const & rhs) const;
    bool operator!=(std::uint32_t const & rhs) const;

    RPCAMCLink & operator++();
    RPCAMCLink operator++(int);
    RPCAMCLink & operator--();
    RPCAMCLink operator--(int);

protected:
    int bf_get(int const min, std::uint32_t const mask, int const pos) const;
    RPCAMCLink & bf_set(int const min, int const max, std::uint32_t const mask, int const pos, int const value);
    std::ostream & bf_stream(std::ostream & ostream, int const min, std::uint32_t const mask, int const pos) const;

protected:
    std::uint32_t id_;

    COND_SERIALIZABLE;
};

std::ostream & operator<<(std::ostream & ostream, RPCAMCLink const & link);

#include "CondFormats/RPCObjects/interface/RPCAMCLink.icc"

#endif // CondFormats_RPCObjects_RPCAMCLink_h
