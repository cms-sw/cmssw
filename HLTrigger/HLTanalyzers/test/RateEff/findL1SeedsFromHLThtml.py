#!/usr/bin/python

import sys,HTMLTableParser

def orderedSet(seq, idfun=None): 
    # order preserving
    if idfun is None:
        def idfun(x): return x
        seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result


l1seeds = []
def main(argv) :
    """
    arguments: infile
    """

    #infname = 'HLT_208_V3.html'
    infname = 'HLT_V9.html'
    alen = len(argv)
    if alen > 0: infname = argv[0]

    f = open(infname,'r')

    p = HTMLTableParser.TableParser()
    p.feed(f.read())
    f.close()

    count = 0
    for l in p.doc:
        for row in l:
            count += 1
            if count == 1: continue
            row0 = row[0]
            row1 = row[1].replace('OR','OR ')
            row = row1.split(' OR ')
            for word in row:
                word.strip()
                #print '"'+word+'"'
                #l1seeds[word] = ''
                l1seeds.append(word)

    #print p.doc # Get to the data

    l = orderedSet(l1seeds)
    for seed in l:
        print seed
    
if __name__ == '__main__' :
    main(sys.argv[1:])
