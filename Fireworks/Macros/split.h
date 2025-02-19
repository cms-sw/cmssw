#ifndef split_h
#define split_h

#include <string>

// template <typename E, typename C>
// size_t split(std::basic_string<E> const& s,
//              C &container,
//              E const delimiter,
//              bool keepBlankFields = true)
//
// Function that splits a string 's' at the occurrences of 'delimiter', and adds the pieces at the end of 'container'.
// // If keepBlankFields is false (the default), consecutive delimiters are treated as a single field separator; otherwise, they produce empty fields.
//
// The function is templated on
//   the character type E (used both for string and delimiter)
//   the container type C (container::value_type must be constructible from std::basic_string<E>)
//
// see: http://www.codeproject.com/KB/stl/Split_string.aspx

template <typename E, typename C>
size_t split(std::basic_string<E> const& s,
             C &container,
             E const delimiter,
             bool keepBlankFields = true)
{
    size_t n = 0;
    typename std::basic_string<E>::const_iterator it = s.begin(), end = s.end(), first;
    for (first = it; it != end; ++it)
    {
        // Examine each character and if it matches the delimiter
        if (delimiter == *it)
        {
            if (keepBlankFields || first != it)
            {
                // extract the current field from the string and
                // append the current field to the given container
                container.push_back(std::basic_string<E>(first, it));
                ++n;

                // skip the delimiter
                first = it + 1;
            }
            else
            {
                ++first;
            }
        }
    }
    if (keepBlankFields || first != it)
    {
        // extract the last field from the string and
        // append the last field to the given container
        container.push_back(std::basic_string<E>(first, it));
        ++n;
    }
    return n;
}

#endif // split_h
