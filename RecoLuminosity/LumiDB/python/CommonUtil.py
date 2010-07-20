'''This module collects some frequently used python functions
'''
def pairwise(lst):
    """
    yield item i and item i+1 in lst. e.g.
    (lst[0], lst[1]), (lst[1], lst[2]), ..., (lst[-1], None)
    
    credit to:
    http://code.activestate.com/recipes/409825-look-ahead-one-item-during-iteration
    """
    if not len(lst): return
    #yield None, lst[0]
    for i in range(len(lst)-1):
        yield lst[i], lst[i+1]
    yield lst[-1], None

def findInList(mylist,element):
    """
    check if an element is in the list
    """
    pos=-1
    try:
        pos=mylist.index(element)
    except ValueError:
        pos=-1
    return pos!=-1

def is_intstr(s):
    """test if a string can be converted to a int
    """
    try:
        int(s)
        return True
    except ValueError:
        return False
def is_floatstr(s):
    """test if a string can be converted to a float
    """
    try:
        float(s)
        return True
    except ValueError:
        return False
def count_dups(l):
    '''report the number of duplicates in a python list
    '''
    from collections import defaultdict
    tally=defaultdict(int)
    for x in l:
        tally[x]+=1
    return tally.items()
if __name__=='__main__':
    a=[1,2,3,4,5]
    for i,j in pairwise(a):
        if j :
            print i,j
    lst = ['I1','I2','I1','I3','I4','I4','I7','I7','I7','I7','I7']
    print count_dups(lst)

