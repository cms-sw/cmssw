#!/usr/bin/env python

import sys

class BookingParams(object):
    bookTransitions = ('CTOR', 'BJ', 'BR')
    def __init__(self, params):
        self.params_ = params
        self.bookLogic_ = {}
        self.postfix_ = ""
        self.forceReset_ = False
        for i in BookingParams.bookTransitions:
            self.bookLogic_[i] = 0

    def doCheck(self, testOnly=True):
        if len(self.params_) < 3:
            print '\n\nMaybe a missing booking directive? (CTOR/Constructor - BJ - BR)?\n\n'
            sys.exit(1)
        if not self.params_[2] in BookingParams.bookTransitions:
            print '\n\nUnknown booking logic.\n'
            print 'Valid values are: [%s, %s, %s]\n\n' % BookingParams.bookTransitions
            sys.exit(1)
        if not testOnly:
            self.bookLogic_[self.params_[2]] = 1
        if len(self.params_) > 3:
            self.checkAdditionalParams_(3)
        if len(self.params_) > 4:
            self.checkAdditionalParams_(4)


    def checkAdditionalParams_(self, num):
        if len(self.params_) > num:
            if self.params_[num] == "MultiThread":
                self.postfix_ = self.params_[num]
            elif self.params_[num] == "ForceReset":
                self.forceReset_ = True
            else:
                print "Expecting MultiThread|ForceReset, got %s" % self.params_[num]
                sys.exit(1)

    def multithread(self):
        return not self.postfix_ == ""

    def mt_postfix(self):
        return self.postfix_

    def forceReset(self):
        return self.forceReset_

    def getBookLogic(self, transition):

        """If transition is the selected choice returns True, False in
        all other cases. An additional check is performed to be sure
        that transition is a valid parameter. """

        self.doCheck(testOnly=True)
        return (self.bookLogic_[transition] == 1)
