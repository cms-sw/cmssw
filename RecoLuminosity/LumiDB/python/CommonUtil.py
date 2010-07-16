'''This module collects some frequently used python functions
'''
def pairwise(lst):
    """
    yield item i and item i+1 in lst. e.g.
    (lst[0], lst[1]), (lst[1], lst[2]), ..., (lst[-1], None)
    
    credit to:
    http://code.activestate.com/recipes/409825-look-ahead-one-item-during-iteration
    """
    if not lst: return
    #yield None, lst[0]
    for i in range(len(lst)-1):
        yield lst[i], lst[i+1]
    yield lst[-1], None
    
if __name__=='__main__':
    a=[1,2,3,4,5]
    for i,j in pairwise(a):
        if j :
            print i,j

