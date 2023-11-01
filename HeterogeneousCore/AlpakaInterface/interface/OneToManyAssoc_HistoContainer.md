# OneToManyAssoc and HistoContainer

## OneToManyAssoc

`OneToManyAssoc*` are containers associating contiguous indices starting from 0 to a variable number of elements in a content array. The associations are first filled up and then queried. They come in 2 flavors: `OneToManyAssocRandom` and `OneToManySeqeuential`. The naming refers to the way they are filled up.

```C++
    template <typename I,    // type stored in the container (usually an index in a vector of the input values)
              int32_t ONES,  // number of "Ones"  +1. If -1 is initialized at runtime using external storage
              int32_t SIZE   // max number of element. If -1 is initialized at runtime using external storage
              >
    class OneToManyAssocBase;
    template <typename I,    // type stored in the container (usually an index in a vector of the input values)
              int32_t ONES,  // number of "Ones"  +1. If -1 is initialized at runtime using external storage
              int32_t SIZE   // max number of element. If -1 is initialized at runtime using external storage
              >
    class OneToManyAssocRandomAccess : public OneToManyAssocBase<I, ONES, SIZE> {...};
    class OneToManyAssocSequential : public OneToManyAssocBase<I, ONES, SIZE> {...};
```

_Data members are_:
 - off[ONES] : array of offsets in the content. An extra offset has to be reserved by the user to record the size of the last serie (sizes are deduced from offset difference). So there are ONES - 1 available slots.
 - content[SIZE] : storage for SIZE elements shared by all the slots. Usually an interger type used as an index to an external storage but could be any type (only tested with intergers so far)

_Filling up goes as_ :
 - `OneToManyAssocRandom` targets unsorted data, which will be assigned to slots on the fly. The filling up is done in 3 phases: a first phase counts the number of elements in each slot, the second (prefix scan) computes the offsets after counting and second pass with the data fills up the sections of the `content[]` array. Counting ad filling up can be done in parallel, one element per thread. The prefix scan is also optimized on GPU.

 [sequence]

 - `OneToManySequential` targets sorted data where parallel threads fill up the association one slots a time. The order/index for each slot is assigned in random order is the threads are competing. The association is filed in one pass. [ TODO: closing? ]

_After filling up, access_:

_Optimizations_:

 - Workspace...