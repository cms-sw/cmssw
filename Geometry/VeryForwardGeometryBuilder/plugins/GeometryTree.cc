/**
 * \file    GeometryTree.cc
 * \author  Tomasz Sodzawiczny (tomasz.sodzawiczny@cern.ch)
 * \date    July 2014
 * \brief   RP Geometry debugging module
 */

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include <iostream>
#include <cstdio>
#include <stack>
#include <string>
#include <vector>

#include <Math/RotationZYX.h>

using namespace edm;
using namespace std;

/* switch between "graphical" tree ouput and basic indentation */
/* could probably be done by cfg.py parameter, don't know how to do it yet */
#define USE_TREE_OUTPUT true 



/**
 * Geometry Tree Printer
 * Prints the RP Geometry hierarchy as a tree
 * \ingroup TotemRPGeometry
 */
class GeometryTree : public edm::EDAnalyzer {
    public:
        explicit GeometryTree(const edm::ParameterSet&) {
        }
        ~GeometryTree() {
        }

    private:
        virtual void beginJob() override {
        }
        virtual void endJob() override {
        }
        virtual void analyze(const edm::Event&, const edm::EventSetup& iSetup) override {

            ESHandle<DetGeomDesc> geometryDescription;
            iSetup.get<VeryForwardMeasuredGeometryRecord>().get(geometryDescription);

            // DFS traversal of Geometry Hierarchy
            stack<Node *> nodeStack;
            nodeStack.push(new Node((DetGeomDesc *)geometryDescription.product(), 0, true, USE_TREE_OUTPUT));
            while (!nodeStack.empty()) {
                Node *node = nodeStack.top();
                nodeStack.pop();

                node->print();

                int childCount = node->desc->components().size();
                int childIndent = node->indent + 1;

                for (int i = 0; i < childCount; ++i) {
                    DetGeomDesc *child = node->desc->components()[i];
                    bool childIsLast = (i == 0) ? true : false;
                    nodeStack.push(new Node(child, childIndent, childIsLast, USE_TREE_OUTPUT));
                }

                delete node;
            }
        }

        /** 
         * Geometry Hierarchy node
         * Used for DFS traversal, keeps tree printing data
         */
        class Node {
            public:
                Node(DetGeomDesc *desc, int indent, bool isLast, bool useTreeOutput = true):
                    desc(desc),
                    indent(indent),
                    isLast(isLast),
                    useTreeOutput(useTreeOutput) {
                }
                /** Actual data */
                DetGeomDesc *desc;
                /** Indent, 0 for the top node, +1 for every next level */
                int indent;
                /** True for every first child put on the stack, false otherwise */
                bool isLast;
                /** Tree output if true, simple indentation otherwise */
                bool useTreeOutput;

                /** 
                 * Prints a line of tree output 
                 * Prints a line formated accordingly to `indent`, `isLast` and `useTreeOutput`.
                 * Do remember to set useTreeOutput he same in all nodes, unpredictable output otherwise.
                 */
                void print(void) {
                    if (useTreeOutput) {
                        // compute indent
                        static vector<char> indentChars;
                        while ((int) indentChars.size() <= indent) {
                            indentChars.push_back(' ');
                        }

                        // print indent
                        for (int i = 0; i < indent - 1; ++i) {
                            printf("%c   ", indentChars[i]);
                        }
                        if (indent > 0) {
                            if (isLast) {
                                printf("`-- ");
                                indentChars[indent-1] = ' ';
                            }
                            else {
                                printf("|-- ");
                                indentChars[indent-1] = '|';
                            }
                        }
                    }
                    else {
                        // print indent
                        for (int i = 0; i < indent; ++i)
                            printf("    ");
                    }

                    // print actual data
                    ROOT::Math::RotationZYX rot;
                    rot = desc->rotation();
                    double rotComponents[3] = {0.,};
                    rot.GetComponents(rotComponents);

                    cout << desc->name().name() << " at " << desc->translation() << " rotated (ZYX) by (";
                    for (int i = 0; i < 3; ++i) {
                        printf("%.2f%c", (rotComponents[i] * 180 / 3.14159), ((i < 2) ? ' ' : ')'));
                    }
                    printf("\n");
                }
        };
};

//define this as a plug-in
DEFINE_FWK_MODULE(GeometryTree);
