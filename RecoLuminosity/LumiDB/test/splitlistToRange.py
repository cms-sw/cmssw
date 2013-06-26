def splitlistToRange(inPut):
    result=[]
    first=inPut[0]
    last=inPut[0]
    result.append([inPut[0]])
    counter=0
    for i in inPut[1:]:
        if i==last+1:
            result[counter].append(i)
        else:
            counter+=1
            result.append([i])
        last=i
    return result
if __name__=='__main__':
    i=[1,2,3,4,5,6,8,9,10]
    print 'input ',i
    isplit=splitlistToRange(i)
    print ['['+str(min(x))+'-'+str(max(x))+']' for x in isplit]
    i=[1,3,5,6,8,9,10,97,100]
    print 'input ',i
    isplit=splitlistToRange(i)
    print ['['+str(min(x))+'-'+str(max(x))+']' for x in isplit]
