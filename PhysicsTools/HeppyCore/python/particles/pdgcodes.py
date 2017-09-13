'''Most tools taken from Rivet

Enumerator:
    nj 	
    nq3 	
    nq2 	
    nq1 	
    nl 	
    nr 	
    n 	
    n8 	
    n9 	
    n10 	
'''

nj, nq3, nq2, nq1, nl, nr, n, n8, n9, n10 = range(1, 11)

def extraBits(pid):
    return abs(pid)/10000000

def digit(loc, pid):
    '''returns the digit at the given location in the pid
            //  PID digits (base 10) are: n nr nl nq1 nq2 nq3 nj
        //  the location enum provides a convenient index into the PID
        int numerator = (int) std::pow(10.0,(loc-1));
        return (abspid(pid)/numerator)%10;
    '''
    # if loc==0 or loc > len(str(pid)):
    #    raise ValueError('wrong location for pid')
    numerator = int( pow(10, loc-1) )
    return (abs(pid)/numerator)%10
    
def fundamentalId(pid):
    '''extract fundamental id if this is a fundamental particle

    In Rivet:
    
    {
        if( extraBits(pid) > 0 ) return 0; 
        if( digit(nq2,pid) == 0 && digit(nq1,pid) == 0) {
            return abspid(pid)%10000;
        } else if( abspid(pid) <= 100 ) {
            return abspid(pid);
        } else {
            return 0;
        }
    }
    '''
    if extraBits(pid) > 0:
        return 0
    if digit(nq2, pid) == 0 and digit(nq1, pid) == 0:
        return abs(pid)%10000
    elif abs(pid) <= 100:
        return abs(pid)
    else:
        return 0

    
def hasBottom(pid):
    '''returns True if it's a composite particle containing a bottom quark
     {
        if( extraBits(pid) > 0 ) { return false; }
        if( fundamentalID(pid) > 0 ) { return false; }
        if( digit(nq3,pid) == 5 || digit(nq2,pid) == 5 || digit(nq1,pid) == 5 ) { return true; }
        return false;
    }
    '''
    if extraBits(pid) > 0:
        return False
    elif fundamentalId(pid) > 0:
        return False
    elif digit(nq3,pid) == 5 or \
        digit(nq2,pid) == 5 or \
        digit(nq1,pid) == 5 :
        return True
    else:
        return False
    
