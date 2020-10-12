webpackHotUpdate_N_E("pages/index",{

/***/ "./components/workspaces/index.tsx":
/*!*****************************************!*\
  !*** ./components/workspaces/index.tsx ***!
  \*****************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/regenerator */ "./node_modules/@babel/runtime/regenerator/index.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/asyncToGenerator */ "./node_modules/@babel/runtime/helpers/esm/asyncToGenerator.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _workspaces_offline__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../workspaces/offline */ "./workspaces/offline.ts");
/* harmony import */ var _workspaces_online__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../workspaces/online */ "./workspaces/online.ts");
/* harmony import */ var _viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../viewDetailsMenu/styledComponents */ "./components/viewDetailsMenu/styledComponents.tsx");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! antd/lib/form/Form */ "./node_modules/antd/lib/form/Form.js");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8___default = /*#__PURE__*/__webpack_require__.n(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8__);
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_10___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_10__);
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ./utils */ "./components/workspaces/utils.ts");
/* harmony import */ var _styles_theme__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../styles/theme */ "./styles/theme.ts");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");




var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/workspaces/index.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_3__["createElement"];











var TabPane = antd__WEBPACK_IMPORTED_MODULE_4__["Tabs"].TabPane;

var Workspaces = function Workspaces() {
  _s();

  var workspaces = _config_config__WEBPACK_IMPORTED_MODULE_13__["functions_config"].mode === 'ONLINE' ? _workspaces_online__WEBPACK_IMPORTED_MODULE_6__["workspaces"] : _workspaces_offline__WEBPACK_IMPORTED_MODULE_5__["workspaces"];
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_10__["useRouter"])();
  var query = router.query;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_3__["useState"](false),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__["default"])(_React$useState, 2),
      openWorkspaces = _React$useState2[0],
      toggleWorkspaces = _React$useState2[1];

  var _React$useState3 = react__WEBPACK_IMPORTED_MODULE_3__["useState"](query.workspaces),
      _React$useState4 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__["default"])(_React$useState3, 2),
      workspace = _React$useState4[0],
      setWorkspace = _React$useState4[1];

  console.log(workspace); // React.useEffect(() => {
  //   setWorkspace(query.workspace)
  // }, [query.workspace])
  // make a workspace set from context

  return __jsx(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8___default.a, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 37,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_9__["StyledFormItem"], {
    labelcolor: "white",
    label: "Workspace",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 38,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Button"], {
    onClick: function onClick() {
      toggleWorkspaces(!openWorkspaces);
    },
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 39,
      columnNumber: 9
    }
  }, workspace), __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["StyledModal"], {
    title: "Workspaces",
    visible: openWorkspaces,
    onCancel: function onCancel() {
      return toggleWorkspaces(false);
    },
    footer: [__jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_9__["StyledButton"], {
      color: _styles_theme__WEBPACK_IMPORTED_MODULE_12__["theme"].colors.secondary.main,
      background: "white",
      key: "Close",
      onClick: function onClick() {
        return toggleWorkspaces(false);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 52,
        columnNumber: 13
      }
    }, "Close")],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 47,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Tabs"], {
    defaultActiveKey: "1",
    type: "card",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 62,
      columnNumber: 11
    }
  }, workspaces.map(function (workspace) {
    return __jsx(TabPane, {
      key: workspace.label,
      tab: workspace.label,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 64,
        columnNumber: 15
      }
    }, workspace.workspaces.map(function (subWorkspace) {
      return __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Button"], {
        key: subWorkspace.label,
        type: "link",
        onClick: /*#__PURE__*/Object(_babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__["default"])( /*#__PURE__*/_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.mark(function _callee() {
          return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.wrap(function _callee$(_context) {
            while (1) {
              switch (_context.prev = _context.next) {
                case 0:
                  setWorkspace(subWorkspace.label);
                  toggleWorkspaces(!openWorkspaces); //if workspace is selected, folder_path in query is set to ''. Then we can regonize
                  //that workspace is selected, and wee need to filter the forst layer of folders.

                  _context.next = 4;
                  return Object(_utils__WEBPACK_IMPORTED_MODULE_11__["setWorkspaceToQuery"])(query, subWorkspace.label);

                case 4:
                case "end":
                  return _context.stop();
              }
            }
          }, _callee);
        })),
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 66,
          columnNumber: 19
        }
      }, subWorkspace.label);
    }));
  })))));
};

_s(Workspaces, "6PCNgLfayYCNiSuUHWRdRbqMTfw=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_10__["useRouter"]];
});

_c = Workspaces;
/* harmony default export */ __webpack_exports__["default"] = (Workspaces);

var _c;

$RefreshReg$(_c, "Workspaces");

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy93b3Jrc3BhY2VzL2luZGV4LnRzeCJdLCJuYW1lcyI6WyJUYWJQYW5lIiwiVGFicyIsIldvcmtzcGFjZXMiLCJ3b3Jrc3BhY2VzIiwiZnVuY3Rpb25zX2NvbmZpZyIsIm1vZGUiLCJvbmxpbmVXb3Jrc3BhY2UiLCJvZmZsaW5lV29yc2twYWNlIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJSZWFjdCIsIm9wZW5Xb3Jrc3BhY2VzIiwidG9nZ2xlV29ya3NwYWNlcyIsIndvcmtzcGFjZSIsInNldFdvcmtzcGFjZSIsImNvbnNvbGUiLCJsb2ciLCJ0aGVtZSIsImNvbG9ycyIsInNlY29uZGFyeSIsIm1haW4iLCJtYXAiLCJsYWJlbCIsInN1YldvcmtzcGFjZSIsInNldFdvcmtzcGFjZVRvUXVlcnkiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBR0E7QUFDQTtJQUVRQSxPLEdBQVlDLHlDLENBQVpELE87O0FBTVIsSUFBTUUsVUFBVSxHQUFHLFNBQWJBLFVBQWEsR0FBTTtBQUFBOztBQUN2QixNQUFNQyxVQUFVLEdBQ2RDLGdFQUFnQixDQUFDQyxJQUFqQixLQUEwQixRQUExQixHQUFxQ0MsNkRBQXJDLEdBQXVEQyw4REFEekQ7QUFFQSxNQUFNQyxNQUFNLEdBQUdDLDhEQUFTLEVBQXhCO0FBQ0EsTUFBTUMsS0FBaUIsR0FBR0YsTUFBTSxDQUFDRSxLQUFqQzs7QUFKdUIsd0JBTW9CQyw4Q0FBQSxDQUFlLEtBQWYsQ0FOcEI7QUFBQTtBQUFBLE1BTWhCQyxjQU5nQjtBQUFBLE1BTUFDLGdCQU5BOztBQUFBLHlCQU9XRiw4Q0FBQSxDQUFlRCxLQUFLLENBQUNQLFVBQXJCLENBUFg7QUFBQTtBQUFBLE1BT2hCVyxTQVBnQjtBQUFBLE1BT0xDLFlBUEs7O0FBU3ZCQyxTQUFPLENBQUNDLEdBQVIsQ0FBWUgsU0FBWixFQVR1QixDQVV2QjtBQUNBO0FBQ0E7QUFDQTs7QUFDQSxTQUNFLE1BQUMseURBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsZ0VBQUQ7QUFBZ0IsY0FBVSxFQUFDLE9BQTNCO0FBQW1DLFNBQUssRUFBQyxXQUF6QztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywyQ0FBRDtBQUNFLFdBQU8sRUFBRSxtQkFBTTtBQUNiRCxzQkFBZ0IsQ0FBQyxDQUFDRCxjQUFGLENBQWhCO0FBQ0QsS0FISDtBQUlFLFFBQUksRUFBQyxNQUpQO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FNR0UsU0FOSCxDQURGLEVBU0UsTUFBQyw2RUFBRDtBQUNFLFNBQUssRUFBQyxZQURSO0FBRUUsV0FBTyxFQUFFRixjQUZYO0FBR0UsWUFBUSxFQUFFO0FBQUEsYUFBTUMsZ0JBQWdCLENBQUMsS0FBRCxDQUF0QjtBQUFBLEtBSFo7QUFJRSxVQUFNLEVBQUUsQ0FDTixNQUFDLDhEQUFEO0FBQ0UsV0FBSyxFQUFFSyxvREFBSyxDQUFDQyxNQUFOLENBQWFDLFNBQWIsQ0FBdUJDLElBRGhDO0FBRUUsZ0JBQVUsRUFBQyxPQUZiO0FBR0UsU0FBRyxFQUFDLE9BSE47QUFJRSxhQUFPLEVBQUU7QUFBQSxlQUFNUixnQkFBZ0IsQ0FBQyxLQUFELENBQXRCO0FBQUEsT0FKWDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLGVBRE0sQ0FKVjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBZUUsTUFBQyx5Q0FBRDtBQUFNLG9CQUFnQixFQUFDLEdBQXZCO0FBQTJCLFFBQUksRUFBQyxNQUFoQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0dWLFVBQVUsQ0FBQ21CLEdBQVgsQ0FBZSxVQUFDUixTQUFEO0FBQUEsV0FDZCxNQUFDLE9BQUQ7QUFBUyxTQUFHLEVBQUVBLFNBQVMsQ0FBQ1MsS0FBeEI7QUFBK0IsU0FBRyxFQUFFVCxTQUFTLENBQUNTLEtBQTlDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FDR1QsU0FBUyxDQUFDWCxVQUFWLENBQXFCbUIsR0FBckIsQ0FBeUIsVUFBQ0UsWUFBRDtBQUFBLGFBQ3hCLE1BQUMsMkNBQUQ7QUFDRSxXQUFHLEVBQUVBLFlBQVksQ0FBQ0QsS0FEcEI7QUFFRSxZQUFJLEVBQUMsTUFGUDtBQUdFLGVBQU8sZ01BQUU7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNQUiw4QkFBWSxDQUFDUyxZQUFZLENBQUNELEtBQWQsQ0FBWjtBQUNBVixrQ0FBZ0IsQ0FBQyxDQUFDRCxjQUFGLENBQWhCLENBRk8sQ0FHUDtBQUNBOztBQUpPO0FBQUEseUJBS0RhLG1FQUFtQixDQUFDZixLQUFELEVBQVFjLFlBQVksQ0FBQ0QsS0FBckIsQ0FMbEI7O0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsU0FBRixFQUhUO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsU0FXR0MsWUFBWSxDQUFDRCxLQVhoQixDQUR3QjtBQUFBLEtBQXpCLENBREgsQ0FEYztBQUFBLEdBQWYsQ0FESCxDQWZGLENBVEYsQ0FERixDQURGO0FBbURELENBakVEOztHQUFNckIsVTtVQUdXTyxzRDs7O0tBSFhQLFU7QUFtRVNBLHlFQUFmIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjgzOWM3Mzc0Mzc4YTQzZmYzMzE5LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XG5pbXBvcnQgeyBUYWJzLCBCdXR0b24gfSBmcm9tICdhbnRkJztcblxuaW1wb3J0IHsgd29ya3NwYWNlcyBhcyBvZmZsaW5lV29yc2twYWNlIH0gZnJvbSAnLi4vLi4vd29ya3NwYWNlcy9vZmZsaW5lJztcbmltcG9ydCB7IHdvcmtzcGFjZXMgYXMgb25saW5lV29ya3NwYWNlIH0gZnJvbSAnLi4vLi4vd29ya3NwYWNlcy9vbmxpbmUnO1xuaW1wb3J0IHsgU3R5bGVkTW9kYWwgfSBmcm9tICcuLi92aWV3RGV0YWlsc01lbnUvc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgRm9ybSBmcm9tICdhbnRkL2xpYi9mb3JtL0Zvcm0nO1xuaW1wb3J0IHsgU3R5bGVkRm9ybUl0ZW0sIFN0eWxlZEJ1dHRvbiB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xuaW1wb3J0IHsgc2V0V29ya3NwYWNlVG9RdWVyeSB9IGZyb20gJy4vdXRpbHMnO1xuaW1wb3J0IHsgUXVlcnlQcm9wcyB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcbmltcG9ydCB7IHVzZUNoYW5nZVJvdXRlciB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZUNoYW5nZVJvdXRlcic7XG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uL3N0eWxlcy90aGVtZSc7XG5pbXBvcnQgeyBmdW5jdGlvbnNfY29uZmlnIH0gZnJvbSAnLi4vLi4vY29uZmlnL2NvbmZpZyc7XG5cbmNvbnN0IHsgVGFiUGFuZSB9ID0gVGFicztcblxuaW50ZXJmYWNlIFdvcnNwYWNlUHJvcHMge1xuICBsYWJlbDogc3RyaW5nO1xuICB3b3Jrc3BhY2VzOiBhbnk7XG59XG5jb25zdCBXb3Jrc3BhY2VzID0gKCkgPT4ge1xuICBjb25zdCB3b3Jrc3BhY2VzID1cbiAgICBmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnID8gb25saW5lV29ya3NwYWNlIDogb2ZmbGluZVdvcnNrcGFjZTtcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xuXG4gIGNvbnN0IFtvcGVuV29ya3NwYWNlcywgdG9nZ2xlV29ya3NwYWNlc10gPSBSZWFjdC51c2VTdGF0ZShmYWxzZSk7XG4gIGNvbnN0IFt3b3Jrc3BhY2UsIHNldFdvcmtzcGFjZV0gPSBSZWFjdC51c2VTdGF0ZShxdWVyeS53b3Jrc3BhY2VzKTtcblxuICBjb25zb2xlLmxvZyh3b3Jrc3BhY2UpXG4gIC8vIFJlYWN0LnVzZUVmZmVjdCgoKSA9PiB7XG4gIC8vICAgc2V0V29ya3NwYWNlKHF1ZXJ5LndvcmtzcGFjZSlcbiAgLy8gfSwgW3F1ZXJ5LndvcmtzcGFjZV0pXG4gIC8vIG1ha2UgYSB3b3Jrc3BhY2Ugc2V0IGZyb20gY29udGV4dFxuICByZXR1cm4gKFxuICAgIDxGb3JtPlxuICAgICAgPFN0eWxlZEZvcm1JdGVtIGxhYmVsY29sb3I9XCJ3aGl0ZVwiIGxhYmVsPVwiV29ya3NwYWNlXCI+XG4gICAgICAgIDxCdXR0b25cbiAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XG4gICAgICAgICAgICB0b2dnbGVXb3Jrc3BhY2VzKCFvcGVuV29ya3NwYWNlcyk7XG4gICAgICAgICAgfX1cbiAgICAgICAgICB0eXBlPVwibGlua1wiXG4gICAgICAgID5cbiAgICAgICAgICB7d29ya3NwYWNlfVxuICAgICAgICA8L0J1dHRvbj5cbiAgICAgICAgPFN0eWxlZE1vZGFsXG4gICAgICAgICAgdGl0bGU9XCJXb3Jrc3BhY2VzXCJcbiAgICAgICAgICB2aXNpYmxlPXtvcGVuV29ya3NwYWNlc31cbiAgICAgICAgICBvbkNhbmNlbD17KCkgPT4gdG9nZ2xlV29ya3NwYWNlcyhmYWxzZSl9XG4gICAgICAgICAgZm9vdGVyPXtbXG4gICAgICAgICAgICA8U3R5bGVkQnV0dG9uXG4gICAgICAgICAgICAgIGNvbG9yPXt0aGVtZS5jb2xvcnMuc2Vjb25kYXJ5Lm1haW59XG4gICAgICAgICAgICAgIGJhY2tncm91bmQ9XCJ3aGl0ZVwiXG4gICAgICAgICAgICAgIGtleT1cIkNsb3NlXCJcbiAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4gdG9nZ2xlV29ya3NwYWNlcyhmYWxzZSl9XG4gICAgICAgICAgICA+XG4gICAgICAgICAgICAgIENsb3NlXG4gICAgICAgICAgICA8L1N0eWxlZEJ1dHRvbj4sXG4gICAgICAgICAgXX1cbiAgICAgICAgPlxuICAgICAgICAgIDxUYWJzIGRlZmF1bHRBY3RpdmVLZXk9XCIxXCIgdHlwZT1cImNhcmRcIj5cbiAgICAgICAgICAgIHt3b3Jrc3BhY2VzLm1hcCgod29ya3NwYWNlOiBXb3JzcGFjZVByb3BzKSA9PiAoXG4gICAgICAgICAgICAgIDxUYWJQYW5lIGtleT17d29ya3NwYWNlLmxhYmVsfSB0YWI9e3dvcmtzcGFjZS5sYWJlbH0+XG4gICAgICAgICAgICAgICAge3dvcmtzcGFjZS53b3Jrc3BhY2VzLm1hcCgoc3ViV29ya3NwYWNlOiBhbnkpID0+IChcbiAgICAgICAgICAgICAgICAgIDxCdXR0b25cbiAgICAgICAgICAgICAgICAgICAga2V5PXtzdWJXb3Jrc3BhY2UubGFiZWx9XG4gICAgICAgICAgICAgICAgICAgIHR5cGU9XCJsaW5rXCJcbiAgICAgICAgICAgICAgICAgICAgb25DbGljaz17YXN5bmMgKCkgPT4ge1xuICAgICAgICAgICAgICAgICAgICAgIHNldFdvcmtzcGFjZShzdWJXb3Jrc3BhY2UubGFiZWwpO1xuICAgICAgICAgICAgICAgICAgICAgIHRvZ2dsZVdvcmtzcGFjZXMoIW9wZW5Xb3Jrc3BhY2VzKTtcbiAgICAgICAgICAgICAgICAgICAgICAvL2lmIHdvcmtzcGFjZSBpcyBzZWxlY3RlZCwgZm9sZGVyX3BhdGggaW4gcXVlcnkgaXMgc2V0IHRvICcnLiBUaGVuIHdlIGNhbiByZWdvbml6ZVxuICAgICAgICAgICAgICAgICAgICAgIC8vdGhhdCB3b3Jrc3BhY2UgaXMgc2VsZWN0ZWQsIGFuZCB3ZWUgbmVlZCB0byBmaWx0ZXIgdGhlIGZvcnN0IGxheWVyIG9mIGZvbGRlcnMuXG4gICAgICAgICAgICAgICAgICAgICAgYXdhaXQgc2V0V29ya3NwYWNlVG9RdWVyeShxdWVyeSwgc3ViV29ya3NwYWNlLmxhYmVsKTtcbiAgICAgICAgICAgICAgICAgICAgfX1cbiAgICAgICAgICAgICAgICAgID5cbiAgICAgICAgICAgICAgICAgICAge3N1YldvcmtzcGFjZS5sYWJlbH1cbiAgICAgICAgICAgICAgICAgIDwvQnV0dG9uPlxuICAgICAgICAgICAgICAgICkpfVxuICAgICAgICAgICAgICA8L1RhYlBhbmU+XG4gICAgICAgICAgICApKX1cbiAgICAgICAgICA8L1RhYnM+XG4gICAgICAgIDwvU3R5bGVkTW9kYWw+XG4gICAgICA8L1N0eWxlZEZvcm1JdGVtPlxuICAgIDwvRm9ybT5cbiAgKTtcbn07XG5cbmV4cG9ydCBkZWZhdWx0IFdvcmtzcGFjZXM7XG4iXSwic291cmNlUm9vdCI6IiJ9